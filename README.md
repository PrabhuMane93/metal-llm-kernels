# metal-llm-kernels

![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%7C%20Metal%203-lightgrey?logo=apple&logoColor=white)

Hand-written Metal compute kernels for LLM inference on Apple Silicon, with a Rust host (`objc2-metal`) and kernels written directly in Metal Shading Language — no MPS, no shortcuts through Apple's own matmul library.

Each kernel is grounded in a real operation from published LLM-inference-runtime research (prefill/decode kernel design, fused dequantization, tiled attention), not a generic GPU tutorial exercise. The Q4 quantization scheme specifically (block size, asymmetric scale/bias, lane-strided packing) is verified directly against the target runtime's own open-source `base-convert` component, not assumed from its papers. Every kernel's output is checked against an independent NumPy reference computed from the exact same input data — never a hand-typed "expected value."

## Project structure

This is a Cargo workspace — one member crate per kernel family, plus a unified Python benchmark harness that's the single source of truth for test data, reference computation, and correctness/timing reporting across all of them:

```
metal-llm-kernels/
├── Cargo.toml              # workspace manifest — member list + shared dependency versions
├── Cargo.lock               # one shared lockfile across all kernel crates
├── requirements.txt          # shared Python deps (numpy, safetensors) for bench/
├── .gitignore
├── README.md                 # this file
│
├── gemv/                     # GEMV 1 — naive
│   ├── Cargo.toml
│   ├── src/main.rs, src/gemv.metal
│   └── README.md
│
├── tiled_gemv/                # GEMV 2 — tiled, threadgroup-cached x
│   ├── Cargo.toml
│   └── src/main.rs, src/gemv_tiled.metal
│
├── fused-gemv-gemm/           # GEMV 3 + GEMM, both fused Q4 dequant
│   ├── Cargo.toml
│   ├── src/lib.rs             # shared: quantize_q4, load_fp32_w, write_base_file, quantize_safetensors_to_base
│   ├── src/gemv_q4_fused.metal, src/gemm_q4_fused.metal
│   └── src/bin/gemv.rs, src/bin/gemm.rs   # two symmetric binaries sharing one library
│
├── bench/                     # unified harness — single source of truth for every op above
│   ├── generators.py          # materializes every op's test inputs, once, upfront
│   ├── reference_ops.py       # NumPy references, including the block-64 Q4 quantized ones
│   ├── compare.py             # shape-agnostic diff report (vector or matrix output)
│   ├── run_bench.py           # orchestrates: load inputs -> numpy ref -> cargo run --release -> diff -> report
│   └── data/                  # generated *.safetensors / *.base files, gitignored
│
└── tiled-attention/            # not started yet
```

## Kernel roadmap

| Kernel | Status | What it demonstrates |
|---|---|---|
| GEMV (naive) | ✅ done | Matrix × vector — the most-executed op in decode, memory-bandwidth-bound |
| GEMV (tiled) | ✅ done | Threadgroup-memory caching of `x`, cooperative load, correctness-critical barriers |
| GEMV + fused Q4 dequant | ✅ done | Dequantization folded into the inner loop — packed weights never materialized to a full-precision array; block-64 asymmetric scheme, verified against the real `base-convert` source |
| GEMM + fused Q4 dequant | ✅ done | Same fused-dequant scheme, batched across tokens in one dispatch — prefill, compute-bound |
| Fused SiLU | planned | Activation fused into a projection kernel |
| Tiled attention | planned | FlashAttention-style QKᵀ/PV with online softmax |

**Open work:**
- Achieved-memory-bandwidth-vs-theoretical-peak measurement and a roofline plot haven't been done for any kernel yet — timing and correctness are verified, the bandwidth writeup isn't.
- **All numbers measured so far are on an M1 (base, ~68 GB/s peak).** A real M4 Pro run (273 GB/s — the class of hardware the target runtime's own papers benchmark against) is still **pending**. A free run on a borrowed M4 (base, not Pro, ~120 GB/s) is planned first as a cheap sanity check before that.
- Threadgroup-memory tiling and Q4 fusion have never been combined in one kernel, for either GEMV or GEMM.
- GPU profiling (Xcode Metal System Trace / GPU frame capture) hasn't been done on any of these kernels yet.

## Setup

**Prerequisites:** a Mac with Apple Silicon (Metal 3+) and Xcode Command Line Tools (`xcode-select --install` if not already present).

**1. Install Rust**, if not already installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**2. Create and activate a Python virtual environment** (used for test-data generation and reference-checking, not for anything performance-critical):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**4. Build the workspace:**
```bash
cargo build
```

Rust dependencies are pulled and built automatically from `Cargo.toml` — no separate install step beyond having `cargo` itself.

## Running the benchmark suite

```bash
python3 bench/generators.py   # once, or whenever test inputs need regenerating
python3 bench/run_bench.py    # runs every registered kernel, diffs against NumPy, reports timing
```

This runs all four kernels above (`gemv_naive`, `gemv_tiled`, `gemv_q4_fused`, `gemm_q4_fused`) and prints a pass/fail summary.

**Running one kernel directly**, without the full harness:
```bash
cargo run -p gemv --release -- --input bench/data/gemv_inputs.safetensors --output /tmp/out.safetensors
cargo run -p tiled_gemv --release -- --input bench/data/gemv_inputs.safetensors --output /tmp/out.safetensors
cargo run -p fused-gemv-gemm --bin gemv --release -- --input bench/data/gemv_inputs.safetensors --output /tmp/out.safetensors
cargo run -p fused-gemv-gemm --bin gemm --release -- --input bench/data/gemm_inputs.safetensors --output /tmp/out.safetensors
```

`fused-gemv-gemm` needs `--bin gemv` or `--bin gemm` explicitly — it's the only crate here with more than one binary, since both kernels share the same quantization library.
