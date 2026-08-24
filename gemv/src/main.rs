use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::NonNull;

use safetensors::SafeTensors;

use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLCreateSystemDefaultDevice, MTLDevice, MTLCommandQueue, MTLLibrary,
    MTLBuffer, MTLResourceOptions,
    MTLCommandEncoder, MTLComputeCommandEncoder, MTLCommandBuffer, MTLSize,
};

// MTLCreateSystemDefaultDevice silently returns None unless CoreGraphics is
// actually linked into the binary — depending on objc2-core-graphics alone
// isn't enough if nothing in it is called, so this forces the link directly.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

fn main() {
    // ── Problem size ────────────────────────────────────────────────
    let num_rows: usize = 4096;
    let num_cols: u32 = 4096;
    let num_cols_usize = num_cols as usize;

    // ── 1. SETUP — once per program run ────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source = NSString::from_str(include_str!("gemv.metal"));
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("shader compile failed");
    let function = library
        .newFunctionWithName(&NSString::from_str("gemv_naive"))
        .expect("kernel not found in library");
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

    // Both W and x come from the same real weight file now — load once, keep the
    // backing bytes alive as long as we need to read from them.
    // CARGO_MANIFEST_DIR is this crate's own directory, resolved at compile time —
    // so this works regardless of what directory `cargo run` is invoked from.
    let inputs_path = concat!(env!("CARGO_MANIFEST_DIR"), "/gemv_inputs.safetensors");
    let file_bytes = std::fs::read(inputs_path)
        .expect("failed to read gemv_inputs.safetensors — run `python3 gen_weights.py` first");
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

    // contents() returns NonNull<c_void> — .as_ptr() unwraps it to a plain raw pointer.
    // Both copies below are the same pattern: raw tensor bytes straight into GPU-shared
    // memory, no intermediate Vec<f32> on the Rust side for either W or x.
    unsafe {
        let w_ptr = w_buffer.contents().as_ptr() as *mut u8;
        let w_bytes = std::slice::from_raw_parts_mut(w_ptr, w_tensor.data().len());
        w_bytes.copy_from_slice(w_tensor.data());

        let x_ptr = x_buffer.contents().as_ptr() as *mut u8;
        let x_bytes = std::slice::from_raw_parts_mut(x_ptr, x_tensor.data().len());
        x_bytes.copy_from_slice(x_tensor.data());
    }

    // ── 3. DISPATCH ─────────────────────────────────────────────────
    let cmd_buffer = queue.commandBuffer().expect("no command buffer");
    let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");

    encoder.setComputePipelineState(&pipeline);
    let num_cols_ptr: NonNull<c_void> = NonNull::from(&num_cols).cast();
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&w_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&x_buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&y_buffer), 0, 2);
        encoder.setBytes_length_atIndex(num_cols_ptr, size_of::<u32>() as NSUInteger, 3);
    }

    let threadgroups = MTLSize { width: num_rows / 256, height: 1, depth: 1 };
    let threads_per_threadgroup = MTLSize { width: 256, height: 1, depth: 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);

    encoder.endEncoding();
    cmd_buffer.commit();
    cmd_buffer.waitUntilCompleted();

    // ── 4. READ BACK ────────────────────────────────────────────────
    let result: &[f32] = unsafe {
        let result_ptr = y_buffer.contents().as_ptr() as *const f32;
        std::slice::from_raw_parts(result_ptr, num_rows)
    };

    println!("y[0] = {}", result[0]);
    println!("y[1] = {}", result[1]);
    println!("y[{}] = {}", num_rows - 1, result[num_rows - 1]);
}