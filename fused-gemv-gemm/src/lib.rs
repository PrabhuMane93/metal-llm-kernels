//! Block-64, asymmetric Q4 quantization (mirrors BaseRT's
//! base-convert/crates/base-quant/src/base_q4.rs) plus a `.base`-style
//! file writer for the result. Exposed as plain functions taking explicit
//! path arguments, so the fused-dequant GEMV binary can call straight into
//! this instead of shelling out to a separate CLI step.

use std::fs;
use std::path::Path;

use half::f16;
use safetensors::SafeTensors;

pub const GROUP_SIZE: usize = 64;
pub const LANES_PER_WORD: usize = 8; // 32 bits / 4 bits per lane
pub const WORDS_PER_GROUP: usize = GROUP_SIZE / LANES_PER_WORD; // 8

pub struct Quantized {
    pub packed_words: Vec<u32>, // WORDS_PER_GROUP per group, all groups concatenated
    pub scales: Vec<f16>,       // one per group
    pub biases: Vec<f16>,       // one per group
}

fn minmax(xs: &[f32]) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in xs {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    (mn, mx)
}

/// Block-64, asymmetric RTN quantization — mirrors
/// base-convert/crates/base-quant/src/base_q4.rs::pack() exactly:
/// scale=(max-min)/15, bias=min, both round-tripped through f16 before
/// being used to quantize, so pack-side math matches dequant-side math.
pub fn quantize_q4(weights: &[f32]) -> Quantized {
    assert!(
        weights.len() % GROUP_SIZE == 0,
        "weights.len()={} must be a multiple of GROUP_SIZE={}",
        weights.len(),
        GROUP_SIZE
    );
    let n_groups = weights.len() / GROUP_SIZE;

    let mut packed_words = Vec::with_capacity(n_groups * WORDS_PER_GROUP);
    let mut scales = Vec::with_capacity(n_groups);
    let mut biases = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let group = &weights[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];

        let (mn, mx) = minmax(group);
        let raw_scale = (mx - mn) / 15.0;
        let scale_f32 = if raw_scale == 0.0 { 1.0 } else { raw_scale };

        // Round-trip scale/bias through f16 BEFORE quantizing — pack-side
        // math must use the same rounded value the kernel reads back at
        // dequant time, or pack and dequant silently disagree.
        let scale_h = f16::from_f32(scale_f32);
        let bias_h = f16::from_f32(mn);
        let scale = scale_h.to_f32();
        let bias = bias_h.to_f32();
        scales.push(scale_h);
        biases.push(bias_h);

        let inv_scale = 1.0 / scale;
        let mut qs = [0u32; GROUP_SIZE];
        for (i, &val) in group.iter().enumerate() {
            qs[i] = ((val - bias) * inv_scale).round().clamp(0.0, 15.0) as u32;
        }

        // Pack 8 lanes per word, lane i at bit i*4 — metal_lane_strided_q4.
        for w in 0..WORDS_PER_GROUP {
            let mut word: u32 = 0;
            for lane in 0..LANES_PER_WORD {
                word |= qs[w * LANES_PER_WORD + lane] << (lane * 4);
            }
            packed_words.push(word);
        }
    }

    Quantized {
        packed_words,
        scales,
        biases,
    }
}

fn f32_vec_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load a named FP32 weight tensor out of a safetensors file, as a flat
/// row-major `Vec<f32>` of length `num_rows * num_cols`. `tensor_name` is
/// parameterized (rather than hardcoded to "W") since a single input file
/// can hold more than one weight matrix — e.g. `silu_inputs.safetensors`
/// holds both `W_gate` and `W_up`.
pub fn load_fp32_w(
    input_path: &str,
    num_rows: usize,
    num_cols: usize,
    tensor_name: &str,
) -> Vec<f32> {
    let file_bytes =
        fs::read(input_path).unwrap_or_else(|e| panic!("failed to read {input_path}: {e}"));
    let tensors =
        SafeTensors::deserialize(&file_bytes).expect("failed to parse safetensors file");
    let w_tensor = tensors
        .tensor(tensor_name)
        .unwrap_or_else(|_| panic!("tensor \"{tensor_name}\" not found in input file"));
    assert_eq!(
        w_tensor.data().len(),
        num_rows * num_cols * 4,
        "{tensor_name} byte length mismatch — expected a {num_rows}x{num_cols} f32 matrix"
    );
    f32_vec_from_le_bytes(w_tensor.data())
}

/// The `.base`-style file construction mechanism, on its own: given an
/// already-quantized `W`, build the magic/version/header/blob bytes and
/// write them to `output_path`. Matches FORMAT.md's byte layout, minus
/// the 64 KiB zero-copy padding — this project's loader does an explicit
/// read+copy, not mmap, so the page-alignment precondition for
/// `MTLBuffer.makeBufferWithBytesNoCopy` buys nothing yet.
pub fn write_base_file(
    output_path: &str,
    q: &Quantized,
    num_rows: usize,
    num_cols: usize,
    tensor_name: &str,
) {
    let mut packed_bytes = Vec::with_capacity(q.packed_words.len() * 4);
    for w in &q.packed_words {
        packed_bytes.extend_from_slice(&w.to_le_bytes());
    }
    let mut scale_bytes = Vec::with_capacity(q.scales.len() * 2);
    for s in &q.scales {
        scale_bytes.extend_from_slice(&s.to_le_bytes());
    }
    let mut bias_bytes = Vec::with_capacity(q.biases.len() * 2);
    for b in &q.biases {
        bias_bytes.extend_from_slice(&b.to_le_bytes());
    }

    // One TensorEntry, matching real .base's field names (offset/length +
    // scale_offset/scale_length + bias_offset/bias_length), scoped to
    // just this one tensor.
    let offset = 0usize;
    let length = packed_bytes.len();
    let scale_offset = offset + length;
    let scale_length = scale_bytes.len();
    let bias_offset = scale_offset + scale_length;
    let bias_length = bias_bytes.len();

    let header = serde_json::json!({
        "schema": 1,
        "tensors": [
            {
                "name": tensor_name,
                "dtype": "base_q4",
                "shape": [num_rows, num_cols],
                "offset": offset,
                "length": length,
                "scale_offset": scale_offset,
                "scale_length": scale_length,
                "bias_offset": bias_offset,
                "bias_length": bias_length,
                "group_size": GROUP_SIZE,
                "layout": "metal_lane_strided_q4"
            }
        ]
    });
    let header_json = serde_json::to_vec(&header).expect("failed to serialize header");

    let mut out =
        Vec::with_capacity(16 + header_json.len() + length + scale_length + bias_length);
    out.extend_from_slice(b"BASE");
    out.extend_from_slice(&1u32.to_le_bytes()); // format_version
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes()); // header_len
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&packed_bytes);
    out.extend_from_slice(&scale_bytes);
    out.extend_from_slice(&bias_bytes);

    fs::write(Path::new(output_path), &out)
        .unwrap_or_else(|e| panic!("failed to write {output_path}: {e}"));
}

/// Orchestrator: read FP32 `W` from `input_path`, quantize it, write the
/// result as a `.base`-style file at `output_path`. The one function to
/// call from bench / from the fused-GEMV binary to go straight from
/// "plain FP32 safetensors" to "quantized .base file" in one call.
pub fn quantize_safetensors_to_base(
    input_path: &str,
    output_path: &str,
    num_rows: usize,
    num_cols: usize,
    tensor_name: &str,
) {
    let w_data = load_fp32_w(input_path, num_rows, num_cols, tensor_name);
    // 4096 (row length) is a multiple of GROUP_SIZE (64), so blocks never
    // straddle a row boundary — quantizing the flat array group-by-group
    // is already correct, no per-row loop needed.
    let q = quantize_q4(&w_data);
    write_base_file(output_path, &q, num_rows, num_cols, tensor_name);
}
