use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::path::Path;
use std::ptr::NonNull;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use fused_gemv_gemm::{quantize_safetensors_to_base, GROUP_SIZE};

// MTLCreateSystemDefaultDevice silently returns None unless CoreGraphics is
// actually linked into the binary — same fix as every other binary here.
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

/// Read one .base file and return its packed/scale/bias byte regions as
/// owned buffers. Called once for W_gate, once for W_up — same file
/// format GEMV 3/GEMM already use, just read twice since this op has two
/// weight matrices instead of one.
fn load_q4_regions(base_path: &str, expected_name: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let base_bytes =
        std::fs::read(base_path).unwrap_or_else(|e| panic!("failed to read {base_path}: {e}"));
    let (header, blob_start) = read_base_header(&base_bytes);
    let tensor = &header["tensors"][0];
    assert_eq!(tensor["name"].as_str(), Some(expected_name));
    assert_eq!(tensor["dtype"].as_str(), Some("base_q4"));
    assert_eq!(tensor["group_size"].as_u64(), Some(GROUP_SIZE as u64));

    let w_offset = tensor["offset"].as_u64().unwrap() as usize;
    let w_length = tensor["length"].as_u64().unwrap() as usize;
    let scale_offset = tensor["scale_offset"].as_u64().unwrap() as usize;
    let scale_length = tensor["scale_length"].as_u64().unwrap() as usize;
    let bias_offset = tensor["bias_offset"].as_u64().unwrap() as usize;
    let bias_length = tensor["bias_length"].as_u64().unwrap() as usize;

    let packed = base_bytes[blob_start + w_offset..blob_start + w_offset + w_length].to_vec();
    let scale =
        base_bytes[blob_start + scale_offset..blob_start + scale_offset + scale_length].to_vec();
    let bias =
        base_bytes[blob_start + bias_offset..blob_start + bias_offset + bias_length].to_vec();
    (packed, scale, bias)
}

fn make_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let buffer = device
        .newBufferWithLength_options(bytes.len() as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("buffer alloc failed");
    unsafe {
        let ptr = buffer.contents().as_ptr() as *mut u8;
        std::slice::from_raw_parts_mut(ptr, bytes.len()).copy_from_slice(bytes);
    }
    buffer
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path =
        parse_arg(&args, "--input").expect("missing --input <fp32 safetensors path (W_gate/W_up/x)>");
    let output_path = parse_arg(&args, "--output").expect("missing --output <path>");

    let num_rows: usize = 4096;
    let num_cols: usize = 4096;

    // ── 0. QUANTIZE — W_gate and W_up each get their own .base file ───
    let gate_base_path = input_path.replace(".safetensors", "_gate_q4.base");
    let up_base_path = input_path.replace(".safetensors", "_up_q4.base");
    quantize_safetensors_to_base(&input_path, &gate_base_path, num_rows, num_cols, "W_gate");
    quantize_safetensors_to_base(&input_path, &up_base_path, num_rows, num_cols, "W_up");

    // ── 1. SETUP — once per program run ────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source = NSString::from_str(include_str!("gemv_silu_fused.metal"));
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("shader compile failed");
    let function = library
        .newFunctionWithName(&NSString::from_str("gemv_silu_fused"))
        .expect("gemv_silu_fused kernel not found in library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation failed");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. LOAD both .base files ───────────────────────────────────
    let (gate_packed, gate_scale, gate_bias) = load_q4_regions(&gate_base_path, "W_gate");
    let (up_packed, up_scale, up_bias) = load_q4_regions(&up_base_path, "W_up");

    // ── 3. BUFFERS ──────────────────────────────────────────────────
    let w_gate_packed_buffer = make_buffer(&device, &gate_packed);
    let gate_scales_buffer = make_buffer(&device, &gate_scale);
    let gate_biases_buffer = make_buffer(&device, &gate_bias);
    let w_up_packed_buffer = make_buffer(&device, &up_packed);
    let up_scales_buffer = make_buffer(&device, &up_scale);
    let up_biases_buffer = make_buffer(&device, &up_bias);

    // x is still plain FP32 — an activation, not a model weight, so it
    // comes straight from the original safetensors file, not the .base.
    let input_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("failed to read {input_path}: {e}"));
    let input_tensors =
        SafeTensors::deserialize(&input_bytes).expect("failed to parse input safetensors file");
    let x_tensor = input_tensors.tensor("x").expect("tensor \"x\" not found in input file");
    assert_eq!(
        x_tensor.data().len(),
        num_cols * size_of::<f32>(),
        "x byte length mismatch"
    );
    let x_buffer = make_buffer(&device, x_tensor.data());

    let h_buffer = device
        .newBufferWithLength_options(
            (num_rows * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("h_buffer alloc failed");

    // ── 4. DISPATCH ─────────────────────────────────────────────────
    let num_cols_u32 = num_cols as u32;
    let num_cols_ptr: NonNull<c_void> = NonNull::from(&num_cols_u32).cast();
    let threadgroups = MTLSize {
        width: num_rows / 256,
        height: 1,
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
            encoder.setBuffer_offset_atIndex(Some(&w_gate_packed_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&gate_scales_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&gate_biases_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&w_up_packed_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&up_scales_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&up_biases_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&x_buffer), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&h_buffer), 0, 7);
            encoder.setBytes_length_atIndex(num_cols_ptr, size_of::<u32>() as NSUInteger, 8);
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
        let result_ptr = h_buffer.contents().as_ptr() as *const f32;
        std::slice::from_raw_parts(result_ptr, num_rows)
    };

    println!("h[0] = {}", result[0]);
    println!("h[1] = {}", result[1]);
    println!("h[{}] = {}", num_rows - 1, result[num_rows - 1]);
    println!("TIMING_MS: {avg_ms:.4}");

    // ── 6. WRITE FULL OUTPUT ────────────────────────────────────────
    let h_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(result.as_ptr() as *const u8, result.len() * size_of::<f32>())
    };
    let h_view =
        TensorView::new(Dtype::F32, vec![num_rows], h_bytes).expect("failed to build TensorView for h");
    let mut out_tensors = HashMap::new();
    out_tensors.insert("h".to_string(), h_view);
    safetensors::tensor::serialize_to_file(&out_tensors, None, Path::new(&output_path))
        .expect("failed to write output safetensors file");
}
