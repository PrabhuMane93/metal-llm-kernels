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

    // Must match the .metal file's SEQ_LEN/D_HEAD/BR/BC exactly — threadgroup
    // array sizes there are compile-time constants, not driven by these.
    let seq_len: usize = 128;
    let d_head: usize = 128;
    let br: usize = 4;
    assert_eq!(seq_len % br, 0, "seq_len must be a multiple of BR — no partial-block handling yet");
    let scale: f32 = 1.0 / (d_head as f32).sqrt();

    // ── 1. SETUP — once per program run ────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source = NSString::from_str(include_str!("flash_attention.metal"));
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("shader compile failed");
    let function = library
        .newFunctionWithName(&NSString::from_str("flash_attention"))
        .expect("flash_attention kernel not found in library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation failed");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. LOAD Q, K, V — single head, (seq_len, d_head) each, same input
    // file as attention_naive (Rung A) — both compute the exact same O. ──
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
    let o_buffer = empty_buffer(&device, seq_len * d_head * size_of::<f32>());

    let scale_ptr: NonNull<c_void> = NonNull::from(&scale).cast();

    // ── 4. DISPATCH — one kernel, one threadgroup per Q row-block. Every
    // threadgroup loops over every K/V block internally (the online-softmax
    // running state lives in threadgroup memory + per-thread registers, so
    // it has to stay live across that whole loop inside one dispatch — it
    // can't be split into separate encoder stages the way Rung A was). ───
    let threadgroups = MTLSize {
        width: seq_len / br,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: br * d_head,
        height: 1,
        depth: 1,
    };

    let run_dispatch = || {
        let cmd_buffer = queue.commandBuffer().expect("no command buffer");
        let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&q_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&k_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&v_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&o_buffer), 0, 3);
            encoder.setBytes_length_atIndex(scale_ptr, size_of::<f32>() as NSUInteger, 4);
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
