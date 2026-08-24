# metal-llm-kernels

![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%7C%20Metal%203-lightgrey?logo=apple&logoColor=white)

Hand-written Metal compute kernels for LLM inference on Apple Silicon, with a Rust host (`objc2-metal`) and kernels written directly in Metal Shading Language — no MPS, no shortcuts through Apple's own matmul library.

Each kernel is grounded in a real operation from published LLM-inference-runtime research (prefill/decode kernel design, fused dequantization, tiled attention), not a generic GPU tutorial exercise. Every kernel's output is checked against an independent NumPy reference computed from the exact same input data — never a hand-typed "expected value."

## Project structure

This is a Cargo workspace — one member crate per kernel, so each stage of the curriculum below is buildable, runnable, and reviewable on its own:

```
metal-llm-kernels/
├── Cargo.toml           # workspace manifest — member list + shared dependency versions
├── Cargo.lock            # one shared lockfile across all kernel crates
├── requirements.txt       # shared Python deps (numpy, safetensors) for every kernel's verification scripts
├── .gitignore
├── README.md              # this file
│
├── gemv/                  # kernel 1 — naive GEMV (done)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs         # host: device/pipeline/buffer setup, dispatch, read-back
│   │   └── gemv.metal      # the kernel itself
│   ├── gen_weights.py       # generates W and x, saved as one .safetensors file
│   ├── verify.py             # independent NumPy reference for correctness checking
│   └── README.md              # kernel-specific writeup and usage
│
├── fused-gemv-gemm/        # kernel 2 — fused Q4 dequant + GEMM (planned)
└── tiled-attention/          # kernel 3 — tiled attention, online softmax (planned)
```

## Kernel roadmap

| Kernel | Status | What it demonstrates |
|---|---|---|
| GEMV (naive) | ✅ done | Matrix × vector — the most-executed op in decode, memory-bandwidth-bound |
| GEMV + fused Q4 dequant | 🔜 next | Dequantization folded into the GEMV inner loop — never materializing full-precision weights |
| GEMM | planned | Matrix × matrix — prefill, compute-bound |
| Fused SiLU | planned | Activation fused into a projection kernel |
| Tiled attention | planned | FlashAttention-style QKᵀ/PV with online softmax |

## Setup

**Prerequisites:** a Mac with Apple Silicon (Metal 3+) and Xcode Command Line Tools (`xcode-select --install` if not already present).

**1. Install Rust**, if not already installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**2. Create and activate a Python virtual environment** (used for weight generation and reference-checking, not for anything performance-critical):
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

## Running a kernel

Each kernel has its own README with the exact run sequence — start with [`gemv/README.md`](gemv/README.md).
