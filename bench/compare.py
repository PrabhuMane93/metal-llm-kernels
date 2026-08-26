"""Shape-agnostic diff logic — works identically for a vector (GEMV) or a
matrix (GEMM) output, since NumPy's elementwise ops don't care about rank.

KL divergence is deliberately not here yet — it measures distance between
probability distributions, not raw numeric arrays. It belongs once there's an
actual distribution to compare (attention weights, logits), not for raw
GEMV/GEMM/dequant output. Stashed for when that stage arrives.
"""

import numpy as np


def diff_report(actual: np.ndarray, expected: np.ndarray) -> dict:
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "max_relative_error": float((diff / (np.abs(expected) + 1e-8)).max()),
    }
