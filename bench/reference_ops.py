"""NumPy/SciPy reference implementations — one function per operation family.

Every function must run and return float32, matching the Metal kernels'
precision. Letting anything silently promote to float64 here (a Python float
literal, a default accumulation dtype) would make the reference *more*
accurate than the kernel, and the diff would then measure "FP32 rounding vs.
an unfairly precise FP64 reference" instead of "kernel bug vs. no bug."
"""

import numpy as np


def gemv_reference(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert W.dtype == np.float32 and x.dtype == np.float32, "inputs must already be float32"
    y = W @ x
    assert y.dtype == np.float32, "matmul silently promoted precision — check input dtypes"
    return y


REFERENCE_OPS = {
    "gemv": gemv_reference,
}
