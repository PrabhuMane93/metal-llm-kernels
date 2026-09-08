"""Materializes ALL operations' input matrices/arrays upfront, once, into bench/data/.

Run standalone before run_bench.py:
    python3 bench/generators.py

Kept separate from run_bench.py so inputs don't get regenerated on every
comparison run — generation and comparison are independently repeatable.
"""

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

DATA_DIR = Path(__file__).parent / "data"


def generate_gemv_inputs() -> Path:
    """W (4096, 4096) and x (4096,) — shared by gemv_naive and gemv_tiled,
    since both compute the exact same y = W @ x."""
    num_rows, num_cols = 4096, 4096
    rng = np.random.default_rng(seed=42)
    W = rng.standard_normal((num_rows, num_cols), dtype=np.float32) * 0.02
    x = rng.standard_normal((num_cols,), dtype=np.float32) * 0.02

    out_path = DATA_DIR / "gemv_inputs.safetensors"
    save_file({"W": W, "x": x}, out_path)
    print(f"wrote {out_path.name} — W {W.shape} {W.dtype}, x {x.shape} {x.dtype}")
    return out_path


def generate_gemm_inputs() -> Path:
    """W (4096, 4096) and x_matrix (128, 4096) — GEMM's batched generalization
    of GEMV's y = W @ x: Y = x_matrix @ W.T, one row of x_matrix per token
    (prefill's stacked hidden states), all rows computed in a single dispatch
    instead of one token at a time. seq_len=128 is a representative short
    prompt length, not a hard constraint of the kernel."""
    num_rows, num_cols = 4096, 4096
    seq_len = 128
    rng = np.random.default_rng(seed=43)
    W = rng.standard_normal((num_rows, num_cols), dtype=np.float32) * 0.02
    x_matrix = rng.standard_normal((seq_len, num_cols), dtype=np.float32) * 0.02

    out_path = DATA_DIR / "gemm_inputs.safetensors"
    save_file({"W": W, "x_matrix": x_matrix}, out_path)
    print(
        f"wrote {out_path.name} — W {W.shape} {W.dtype}, "
        f"x_matrix {x_matrix.shape} {x_matrix.dtype}"
    )
    return out_path


def generate_silu_inputs() -> Path:
    """W_gate (4096, 4096), W_up (4096, 4096), and x (4096,) — fused SiLU's
    decode-shaped inputs: gate = W_gate @ x, up = W_up @ x, both GEMV-shaped
    like gemv_inputs (single-token decode, not gemm_inputs' batched prefill),
    matching this task's single-dispatch, two-accumulator design."""
    num_rows, num_cols = 4096, 4096
    rng = np.random.default_rng(seed=44)
    W_gate = rng.standard_normal((num_rows, num_cols), dtype=np.float32) * 0.02
    W_up = rng.standard_normal((num_rows, num_cols), dtype=np.float32) * 0.02
    x = rng.standard_normal((num_cols,), dtype=np.float32) * 0.02

    out_path = DATA_DIR / "silu_inputs.safetensors"
    save_file({"W_gate": W_gate, "W_up": W_up, "x": x}, out_path)
    print(
        f"wrote {out_path.name} — W_gate {W_gate.shape} {W_gate.dtype}, "
        f"W_up {W_up.shape} {W_up.dtype}, x {x.shape} {x.dtype}"
    )
    return out_path


# One entry per distinct input shape/family — not one per kernel binary.
# gemv_naive and gemv_tiled both consume "gemv_inputs", for example.
GENERATORS = {
    "gemv_inputs": generate_gemv_inputs,
    "gemm_inputs": generate_gemm_inputs,
    "silu_inputs": generate_silu_inputs,
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for generator in GENERATORS.values():
        generator()


if __name__ == "__main__":
    main()
