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


# One entry per distinct input shape/family — not one per kernel binary.
# gemv_naive and gemv_tiled both consume "gemv_inputs", for example.
GENERATORS = {
    "gemv_inputs": generate_gemv_inputs,
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for generator in GENERATORS.values():
        generator()


if __name__ == "__main__":
    main()
