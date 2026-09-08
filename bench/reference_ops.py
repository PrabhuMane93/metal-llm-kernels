"""NumPy/SciPy reference implementations — one function per operation family.

Every function must run and return float32, matching the Metal kernels'
precision. Letting anything silently promote to float64 here (a Python float
literal, a default accumulation dtype) would make the reference *more*
accurate than the kernel, and the diff would then measure "FP32 rounding vs.
an unfairly precise FP64 reference" instead of "kernel bug vs. no bug."
"""

import numpy as np
import torch
import torch.nn.functional as F


def gemv_reference(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert W.dtype == np.float32 and x.dtype == np.float32, "inputs must already be float32"
    y = W @ x
    assert y.dtype == np.float32, "matmul silently promoted precision — check input dtypes"
    return y


def _quantize_dequant_block64(W: np.ndarray, group_size: int = 64) -> np.ndarray:
    """Reproduces gemv_q4_fused/gemm_q4_fused's exact scheme (block-64,
    asymmetric, RTN — scale=(max-min)/15, bias=min, both round-tripped
    through f16 before use, matching base-convert's base_q4.rs) so the Q4
    kernels get compared against the mathematically-correct *quantized*
    result, not the lossless FP32 one. Q4's whole point is to diverge from
    FP32 — the reference has to model that divergence on purpose, or every
    comparison fails on real, expected, non-bug error.
    """
    assert W.dtype == np.float32
    num_rows, num_cols = W.shape
    assert num_cols % group_size == 0, f"num_cols={num_cols} must be a multiple of {group_size}"
    blocks = W.reshape(num_rows, num_cols // group_size, group_size)

    mn = blocks.min(axis=2, keepdims=True)
    mx = blocks.max(axis=2, keepdims=True)
    raw_scale = (mx - mn) / 15.0
    scale = np.where(raw_scale == 0.0, 1.0, raw_scale).astype(np.float32)

    # Round-trip through f16 BEFORE quantizing — pack-side math must match
    # what the kernel reads back at dequant time.
    scale = np.float16(scale).astype(np.float32)
    bias = np.float16(mn).astype(np.float32)

    q = np.clip(np.round((blocks - bias) / scale), 0, 15)
    dequant = (q * scale + bias).astype(np.float32)
    return dequant.reshape(num_rows, num_cols)


def gemv_q4_reference(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert W.dtype == np.float32 and x.dtype == np.float32, "inputs must already be float32"
    W_dequant = _quantize_dequant_block64(W)
    y = W_dequant @ x
    assert y.dtype == np.float32, "matmul silently promoted precision — check input dtypes"
    return y


def gemm_q4_reference(W: np.ndarray, x_matrix: np.ndarray) -> np.ndarray:
    assert W.dtype == np.float32 and x_matrix.dtype == np.float32, "inputs must already be float32"
    W_dequant = _quantize_dequant_block64(W)
    Y = x_matrix @ W_dequant.T
    assert Y.dtype == np.float32, "matmul silently promoted precision — check input dtypes"
    return Y


def _swiglu_torch(W_gate: np.ndarray, W_up: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Shared core for both SwiGLU variants below. SiLU comes from PyTorch's
    F.silu — a real, trusted implementation of z*sigmoid(z) — rather than a
    hand-rolled sigmoid formula, same standard already applied to
    _quantize_dequant_block64 (verified against real base-convert source
    instead of assumed). torch.from_numpy is zero-copy for float32 arrays,
    and torch matmul/F.silu don't silently upcast to float64 the way a
    Python float literal can under NumPy — but the inputs must already be
    float32 going in, checked explicitly since nothing here would catch a
    float64 input by accident.
    """
    W_gate_t = torch.from_numpy(W_gate)
    W_up_t = torch.from_numpy(W_up)
    x_t = torch.from_numpy(x)
    assert (
        W_gate_t.dtype == torch.float32
        and W_up_t.dtype == torch.float32
        and x_t.dtype == torch.float32
    ), "inputs must already be float32"

    gate = W_gate_t @ x_t
    up = W_up_t @ x_t
    h = F.silu(gate) * up
    assert h.dtype == torch.float32, "torch op silently promoted precision — check input dtypes"
    return h.numpy()


def swiglu_reference(W_gate: np.ndarray, W_up: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert (
        W_gate.dtype == np.float32 and W_up.dtype == np.float32 and x.dtype == np.float32
    ), "inputs must already be float32"
    return _swiglu_torch(W_gate, W_up, x)


def swiglu_q4_reference(W_gate: np.ndarray, W_up: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert (
        W_gate.dtype == np.float32 and W_up.dtype == np.float32 and x.dtype == np.float32
    ), "inputs must already be float32"
    W_gate_dequant = _quantize_dequant_block64(W_gate)
    W_up_dequant = _quantize_dequant_block64(W_up)
    return _swiglu_torch(W_gate_dequant, W_up_dequant, x)


REFERENCE_OPS = {
    "gemv": gemv_reference,
    "gemv_q4": gemv_q4_reference,
    "gemm_q4": gemm_q4_reference,
    "swiglu": swiglu_reference,
    "swiglu_q4": swiglu_q4_reference,
}