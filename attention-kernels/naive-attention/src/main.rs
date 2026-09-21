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

fn make_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let buffer = device
        .newBufferWithLength_options(bytes.len() as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("buffer alloc failed");
    unsafe {
        let ptr = buffer.contents().as_ptr() as *mut u8;
        std::slice::from_raw_parts_mut(ptr, bytes.len()).copy_from_slice(bytes);
    }
    buffer
}

fn empty_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    len_bytes: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(len_bytes as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("buffer alloc failed")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path =
        parse_arg(&args, "--input").expect("missing --input <fp32 safetensors path (Q/K/V)>");
    let output_path = parse_arg(&args, "--output").expect("missing --output <path>");

    let seq_len: usize = 128;
    let d_head: usize = 128;
    let scale: f32 = 1.0 / (d_head as f32).sqrt();

    // ── 1. SETUP — once per program run ────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source = NSString::from_str(include_str!("attention_naive.metal"));
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("shader compile failed");

    let make_pipeline = |name: &str| {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .unwrap_or_else(|| panic!("{name} kernel not found in library"));
        device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|_| panic!("pipeline creation failed for {name}"))
    };
    let qk_pipeline = make_pipeline("qk_scaled_matmul");
    let softmax_pipeline = make_pipeline("row_softmax");
    let av_pipeline = make_pipeline("av_matmul");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. LOAD Q, K, V — single head, (seq_len, d_head) each ───────
    let input_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("failed to read {input_path}: {e}"));
    let input_tensors =
        SafeTensors::deserialize(&input_bytes).expect("failed to parse input safetensors file");

    let load_tensor = |name: &str| {
        let t = input_tensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("tensor \"{name}\" not found in input file"));
        assert_eq!(
            t.data().len(),
            seq_len * d_head * size_of::<f32>(),
            "{name} byte length mismatch"
        );
        t.data().to_vec()
    };
    let q_bytes = load_tensor("Q");
    let k_bytes = load_tensor("K");
    let v_bytes = load_tensor("V");

    // ── 3. BUFFERS ──────────────────────────────────────────────────
    let q_buffer = make_buffer(&device, &q_bytes);
    let k_buffer = make_buffer(&device, &k_bytes);
    let v_buffer = make_buffer(&device, &v_bytes);
    // S holds QKᵀ*scale, then gets softmaxed in place into A — one
    // buffer, two names, exactly the "materialize the full row" naive
    // path this kernel is meant to be.
    let s_buffer = empty_buffer(&device, seq_len * seq_len * size_of::<f32>());
    let o_buffer = empty_buffer(&device, seq_len * d_head * size_of::<f32>());

    let seq_len_u32 = seq_len as u32;
    let d_head_u32 = d_head as u32;
    let seq_len_ptr: NonNull<c_void> = NonNull::from(&seq_len_u32).cast();
    let d_head_ptr: NonNull<c_void> = NonNull::from(&d_head_u32).cast();
    let scale_ptr: NonNull<c_void> = NonNull::from(&scale).cast();

    // ── 4. DISPATCH — three stages, one command buffer. Metal's
    // automatic hazard tracking on shared-storage buffers orders these
    // correctly (stage 2 waits for stage 1's writes to S, stage 3 waits
    // for stage 2's) without any manual barriers. ───────────────────
    let tg_2d = MTLSize { width: 16, height: 16, depth: 1 };
    let grid_qk = MTLSize {
        width: seq_len / 16,
        height: seq_len / 16,
        depth: 1,
    };
    let grid_av = MTLSize {
        width: d_head / 16,
        height: seq_len / 16,
        depth: 1,
    };
    let tg_softmax = MTLSize {
        width: seq_len,
        height: 1,
        depth: 1,
    };
    let grid_softmax = MTLSize { width: 1, height: 1, depth: 1 };

    let run_dispatch = || {
        let cmd_buffer = queue.commandBuffer().expect("no command buffer");

        {
            let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");
            encoder.setComputePipelineState(&qk_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&q_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&k_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&s_buffer), 0, 2);
                encoder.setBytes_length_atIndex(seq_len_ptr, size_of::<u32>() as NSUInteger, 3);
                encoder.setBytes_length_atIndex(d_head_ptr, size_of::<u32>() as NSUInteger, 4);
                encoder.setBytes_length_atIndex(scale_ptr, size_of::<f32>() as NSUInteger, 5);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg_2d);
            encoder.endEncoding();
        }
        {
            let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");
            encoder.setComputePipelineState(&softmax_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&s_buffer), 0, 0);
                encoder.setBytes_length_atIndex(seq_len_ptr, size_of::<u32>() as NSUInteger, 1);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_softmax, tg_softmax);
            encoder.endEncoding();
        }
        {
            let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");
            encoder.setComputePipelineState(&av_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&s_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&v_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&o_buffer), 0, 2);
                encoder.setBytes_length_atIndex(seq_len_ptr, size_of::<u32>() as NSUInteger, 3);
                encoder.setBytes_length_atIndex(d_head_ptr, size_of::<u32>() as NSUInteger, 4);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_av, tg_2d);
            encoder.endEncoding();
        }

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
        let result_ptr = o_buffer.contents().as_ptr() as *const f32;
        std::slice::from_raw_parts(result_ptr, seq_len * d_head)
    };

    println!("O[0][0] = {}", result[0]);
    println!("O[0][1] = {}", result[1]);
    println!(
        "O[{}][{}] = {}",
        seq_len - 1,
        d_head - 1,
        result[seq_len * d_head - 1]
    );
    println!("TIMING_MS: {avg_ms:.4}");

    // ── 6. WRITE FULL OUTPUT ────────────────────────────────────────
    let o_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(result.as_ptr() as *const u8, result.len() * size_of::<f32>())
    };
    let o_view = TensorView::new(Dtype::F32, vec![seq_len, d_head], o_bytes)
        .expect("failed to build TensorView for O");
    let mut out_tensors = HashMap::new();
    out_tensors.insert("O".to_string(), o_view);
    safetensors::tensor::serialize_to_file(&out_tensors, None, Path::new(&output_path))
        .expect("failed to write output safetensors file");
}
