from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

num_rows, num_cols = 4096, 4096

rng = np.random.default_rng(seed=42)
W = rng.standard_normal((num_rows, num_cols), dtype=np.float32) * 0.02
x = rng.standard_normal((num_cols,), dtype=np.float32) * 0.02

out_path = Path(__file__).parent / "gemv_inputs.safetensors"
save_file({"W": W, "x": x}, out_path)
print(f"wrote {out_path.name} — W {W.shape} {W.dtype}, x {x.shape} {x.dtype}")
