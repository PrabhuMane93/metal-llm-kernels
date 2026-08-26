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

// MTLCreateSystemDefaultDevice silently returns None unless CoreGraphics is
// actually linked into the binary — depending on objc2-core-graphics alone
// isn't enough if nothing in it is called, so this forces the link directly.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path = parse_arg(&args, "--input").expect("missing --input <path>");
    let output_path = parse_arg(&args, "--output").expect("missing --output <path>");

    // ── Problem size ────────────────────────────────────────────────
    let num_rows: usize = 4096;
    let num_cols: u32 = 4096;
    let num_cols_usize = num_cols as usize;

    // TILE_SIZE in the kernel is fixed at 256, matching threads_per_threadgroup below.
    assert_eq!(
        num_cols % 256,
        0,
        "numCols must be a multiple of TILE_SIZE (256) — no partial-tile handling yet"
    );

    // ── 1. SETUP — once per program run ────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source = NSString::from_str(include_str!("gemv_tiled.metal"));
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("shader compile failed");
    let function = library
        .newFunctionWithName(&NSString::from_str("gemv_tiled"))
        .expect("gemv_tiled kernel not found in library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation failed");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. BUFFERS ──────────────────────────────────────────────────
    let w_buffer = device
        .newBufferWithLength_options(
            (num_rows * num_cols_usize * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("w_buffer alloc failed");
    let x_buffer = device
        .newBufferWithLength_options(
            (num_cols_usize * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("x_buffer alloc failed");
    let y_buffer = device
        .newBufferWithLength_options(
            (num_rows * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("y_buffer alloc failed");

    let file_bytes = std::fs::read(&input_path)
        .unwrap_or_else(|e| panic!("failed to read input safetensors file {input_path}: {e}"));
    let tensors = SafeTensors::deserialize(&file_bytes).expect("failed to parse safetensors file");

    let w_tensor = tensors.tensor("W").expect("tensor \"W\" not found in file");
    assert_eq!(
        w_tensor.data().len(),
        num_rows * num_cols_usize * size_of::<f32>(),
        "W byte length mismatch — file shape doesn't match num_rows/num_cols"
    );
    let x_tensor = tensors.tensor("x").expect("tensor \"x\" not found in file");
    assert_eq!(
        x_tensor.data().len(),
        num_cols_usize * size_of::<f32>(),
        "x byte length mismatch — file shape doesn't match num_cols"
    );

    unsafe {
        let w_ptr = w_buffer.contents().as_ptr() as *mut u8;
        let w_bytes = std::slice::from_raw_parts_mut(w_ptr, w_tensor.data().len());
        w_bytes.copy_from_slice(w_tensor.data());

        let x_ptr = x_buffer.contents().as_ptr() as *mut u8;
        let x_bytes = std::slice::from_raw_parts_mut(x_ptr, x_tensor.data().len());
        x_bytes.copy_from_slice(x_tensor.data());
    }

    // ── 3. DISPATCH ─────────────────────────────────────────────────
    let num_cols_ptr: NonNull<c_void> = NonNull::from(&num_cols).cast();
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
            encoder.setBuffer_offset_atIndex(Some(&w_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&x_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&y_buffer), 0, 2);
            encoder.setBytes_length_atIndex(num_cols_ptr, size_of::<u32>() as NSUInteger, 3);
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

    // ── 4. READ BACK ────────────────────────────────────────────────
    let result: &[f32] = unsafe {
        let result_ptr = y_buffer.contents().as_ptr() as *const f32;
        std::slice::from_raw_parts(result_ptr, num_rows)
    };

    println!("y[0] = {}", result[0]);
    println!("y[1] = {}", result[1]);
    println!("y[{}] = {}", num_rows - 1, result[num_rows - 1]);
    println!("TIMING_MS: {avg_ms:.4}");

    // ── 5. WRITE FULL OUTPUT ────────────────────────────────────────
    let y_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(result.as_ptr() as *const u8, result.len() * size_of::<f32>())
    };
    let y_view =
        TensorView::new(Dtype::F32, vec![num_rows], y_bytes).expect("failed to build TensorView for y");
    let mut out_tensors = HashMap::new();
    out_tensors.insert("y".to_string(), y_view);
    safetensors::tensor::serialize_to_file(&out_tensors, None, Path::new(&output_path))
        .expect("failed to write output safetensors file");
}
