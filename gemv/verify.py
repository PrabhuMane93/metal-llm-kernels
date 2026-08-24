from pathlib import Path

from safetensors.numpy import load_file

num_rows = 4096

in_path = Path(__file__).parent / "gemv_inputs.safetensors"
tensors = load_file(in_path)
W = tensors["W"]
x = tensors["x"]

y = W @ x
print("y[0] =", y[0])
print("y[1] =", y[1])
print(f"y[{num_rows - 1}] =", y[num_rows - 1])
