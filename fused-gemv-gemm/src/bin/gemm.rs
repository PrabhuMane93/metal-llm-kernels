use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::path::Path;
use std::ptr::NonNull;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use fused_gemv_gemm::{quantize_safetensors_to_base, GROUP_SIZE};

// MTLCreateSystemDefaultDevice silently returns None unless CoreGraphics is
// actually linked into the binary — same fix as gemv/tiled_gemv/gemv_q4_fused.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Parse the .base header (magic/version/header_len/header_json) back out.
/// Returns (parsed header JSON, byte offset where weights_blob starts).
fn read_base_header(file_bytes: &[u8]) -> (serde_json::Value, usize) {
    assert_eq!(&file_bytes[0..4], b"BASE", "bad magic");
    let version = u32::from_le_bytes(file_bytes[4..8].try_into().unwrap());
    assert_eq!(version, 1, "unsupported format_version");
    let header_len = u64::from_le_bytes(file_bytes[8..16].try_into().unwrap()) as usize;
    let header_bytes = &file_bytes[16..16 + header_len];
    let header = serde_json::from_slice(header_bytes).expect("bad header json");
    let blob_start = 16 + header_len; // no zero-copy padding in this project's writer
    (header, blob_start)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path =
        parse_arg(&args, "--input").expect("missing --input <fp32 safetensors path>");
    let output_path = parse_arg(&args, "--output").expect("missing --output <path>");

    let num_rows: usize = 4096;
    let num_cols: usize = 4096;
    let seq_len: usize = 128;

    // ── 0. QUANTIZE — produce the .base file from the FP32 input ──────
    // Same quantize_q4/write_base_file as GEMV 3 — Q4 packing doesn't
    // care whether the packed W ends up feeding a GEMV or a GEMM kernel.
    let base_path = input_path.replace(".safetensors", "_q4.base");
    quantize_safetensors_to_base(&input_path, &base_path, num_rows, num_cols);

    // ── 1. SETUP — once per program run ────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source = NSString::from_str(include_str!("../gemm_q4_fused.metal"));
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("shader compile failed");
    let function = library
        .newFunctionWithName(&NSString::from_str("gemm_q4_fused"))
        .expect("gemm_q4_fused kernel not found in library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation failed");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. LOAD .base — parse header, slice packed/scale/bias regions ─
    let base_bytes =
        std::fs::read(&base_path).unwrap_or_else(|e| panic!("failed to read {base_path}: {e}"));
    let (header, blob_start) = read_base_header(&base_bytes);
    let tensor = &header["tensors"][0];
    assert_eq!(tensor["name"].as_str(), Some("W"));
    assert_eq!(tensor["dtype"].as_str(), Some("base_q4"));
    assert_eq!(tensor["group_size"].as_u64(), Some(GROUP_SIZE as u64));

    let w_offset = tensor["offset"].as_u64().unwrap() as usize;
    let w_length = tensor["length"].as_u64().unwrap() as usize;
    let scale_offset = tensor["scale_offset"].as_u64().unwrap() as usize;
    let scale_length = tensor["scale_length"].as_u64().unwrap() as usize;
    let bias_offset = tensor["bias_offset"].as_u64().unwrap() as usize;
    let bias_length = tensor["bias_length"].as_u64().unwrap() as usize;

    let packed_bytes = &base_bytes[blob_start + w_offset..blob_start + w_offset + w_length];
    let scale_bytes =
        &base_bytes[blob_start + scale_offset..blob_start + scale_offset + scale_length];
    let bias_bytes = &base_bytes[blob_start + bias_offset..blob_start + bias_offset + bias_length];

    // ── 3. BUFFERS ──────────────────────────────────────────────────
    let w_packed_buffer = device
        .newBufferWithLength_options(w_length as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("w_packed_buffer alloc failed");
    let scales_buffer = device
        .newBufferWithLength_options(scale_length as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("scales_buffer alloc failed");
    let biases_buffer = device
        .newBufferWithLength_options(bias_length as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("biases_buffer alloc failed");
    let x_buffer = device
        .newBufferWithLength_options(
            (seq_len * num_cols * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("x_buffer alloc failed");
    let y_buffer = device
        .newBufferWithLength_options(
            (seq_len * num_rows * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("y_buffer alloc failed");

    unsafe {
        let w_ptr = w_packed_buffer.contents().as_ptr() as *mut u8;
        std::slice::from_raw_parts_mut(w_ptr, packed_bytes.len()).copy_from_slice(packed_bytes);

        let s_ptr = scales_buffer.contents().as_ptr() as *mut u8;
        std::slice::from_raw_parts_mut(s_ptr, scale_bytes.len()).copy_from_slice(scale_bytes);

        let b_ptr = biases_buffer.contents().as_ptr() as *mut u8;
        std::slice::from_raw_parts_mut(b_ptr, bias_bytes.len()).copy_from_slice(bias_bytes);
    }

    // x_matrix is still plain FP32 — an activation, not a model weight, so
    // it comes straight from the original safetensors file, not the .base.
    let input_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("failed to read {input_path}: {e}"));
    let input_tensors =
        SafeTensors::deserialize(&input_bytes).expect("failed to parse input safetensors file");
    let x_tensor = input_tensors
        .tensor("x_matrix")
        .expect("tensor \"x_matrix\" not found in input file");
    assert_eq!(
        x_tensor.data().len(),
        seq_len * num_cols * size_of::<f32>(),
        "x_matrix byte length mismatch"
    );
    unsafe {
        let x_ptr = x_buffer.contents().as_ptr() as *mut u8;
        std::slice::from_raw_parts_mut(x_ptr, x_tensor.data().len())
            .copy_from_slice(x_tensor.data());
    }

    // ── 4. DISPATCH ─────────────────────────────────────────────────
    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let num_cols_ptr: NonNull<c_void> = NonNull::from(&num_cols_u32).cast();
    let num_rows_ptr: NonNull<c_void> = NonNull::from(&num_rows_u32).cast();
    // width covers rows (256 threads/threadgroup, num_rows/256 threadgroups),
    // height covers tokens — one threadgroup-row per token.
    let threadgroups = MTLSize {
        width: num_rows / 256,
        height: seq_len,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    let run_dispatch = || {
        let cmd_buffer = queue.commandBuffer().expect("no command buffer");
        let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&w_packed_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&scales_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&biases_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&x_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&y_buffer), 0, 4);
            encoder.setBytes_length_atIndex(num_cols_ptr, size_of::<u32>() as NSUInteger, 5);
            encoder.setBytes_length_atIndex(num_rows_ptr, size_of::<u32>() as NSUInteger, 6);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        encoder.endEncoding();

        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();
    };

    // Warmup dispatch (excluded from timing) + repeated timed dispatches.
    let warmup_runs = 1;
    let timed_runs = 20;
    for _ in 0..warmup_runs {
        run_dispatch();
    }
    let start = std::time::Instant::now();
    for _ in 0..timed_runs {
        run_dispatch();
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / timed_runs as f64;

    // ── 5. READ BACK ────────────────────────────────────────────────
    let result: &[f32] = unsafe {
        let result_ptr = y_buffer.contents().as_ptr() as *const f32;
        std::slice::from_raw_parts(result_ptr, seq_len * num_rows)
    };

    println!("Y[0][0] = {}", result[0]);
    println!("Y[0][1] = {}", result[1]);
    println!("Y[{}][{}] = {}", seq_len - 1, num_rows - 1, result[seq_len * num_rows - 1]);
    println!("TIMING_MS: {avg_ms:.4}");

    // ── 6. WRITE FULL OUTPUT ────────────────────────────────────────
    let y_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(result.as_ptr() as *const u8, result.len() * size_of::<f32>())
    };
    let y_view = TensorView::new(Dtype::F32, vec![seq_len, num_rows], y_bytes)
        .expect("failed to build TensorView for Y");
    let mut out_tensors = HashMap::new();
    out_tensors.insert("Y".to_string(), y_view);
    safetensors::tensor::serialize_to_file(&out_tensors, None, Path::new(&output_path))
        .expect("failed to write output safetensors file");
}
