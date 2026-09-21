"""Orchestrator: for each registered operation —
    load already-materialized inputs (generators.py must have run already)
    -> run the numpy reference (timed)
    -> run the rust kernel binary via `cargo run --release` (timed internally,
       by the binary itself: warmup + 20 dispatches, see TIMING_MS in its stdout)
    -> diff the two outputs
    -> report correctness + timing

Run:
    python3 bench/generators.py   # once, or whenever inputs need regenerating
    python3 bench/run_bench.py
"""

import re
import subprocess
import sys
import time
from pathlib import Path

from safetensors.numpy import load_file

from compare import diff_report
from reference_ops import REFERENCE_OPS

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"

TIMING_RE = re.compile(r"TIMING_MS:\s*([\d.]+)")

# One entry per kernel *binary* — gemv_naive and gemv_tiled are two entries
# even though they share the same input file, since they're separate crates
# with separate outputs to compare against the same reference. "bin" is only
# needed for crates with more than one binary (fused-gemv-gemm has gemv AND
# gemm as separate src/bin/*.rs, so `cargo run -p fused-gemv-gemm` alone is
# ambiguous without it). "output_tensor" defaults to "y"; GEMM writes "Y".
OPERATIONS = [
    {
        "name": "gemv_naive",
        "crate": "gemv",
        "input_file": "gemv_inputs.safetensors",
        "output_file": "gemv_naive_output.safetensors",
        "reference_op": "gemv",
        "reference_inputs": ("W", "x"),
    },
    {
        "name": "gemv_tiled",
        "crate": "tiled_gemv",
        "input_file": "gemv_inputs.safetensors",
        "output_file": "gemv_tiled_output.safetensors",
        "reference_op": "gemv",
        "reference_inputs": ("W", "x"),
    },
    {
        "name": "gemv_q4_fused",
        "crate": "fused-gemv-gemm",
        "bin": "gemv",
        "input_file": "gemv_inputs.safetensors",
        "output_file": "gemv_q4_fused_output.safetensors",
        "reference_op": "gemv_q4",
        "reference_inputs": ("W", "x"),
    },
    {
        "name": "gemm_q4_fused",
        "crate": "fused-gemv-gemm",
        "bin": "gemm",
        "input_file": "gemm_inputs.safetensors",
        "output_file": "gemm_q4_fused_output.safetensors",
        "output_tensor": "Y",
        "reference_op": "gemm_q4",
        "reference_inputs": ("W", "x_matrix"),
    },
    {
        "name": "gemv_silu_fused",
        "crate": "fused-silu",
        "input_file": "silu_inputs.safetensors",
        "output_file": "silu_fused_output.safetensors",
        "output_tensor": "h",
        "reference_op": "swiglu_q4",
        "reference_inputs": ("W_gate", "W_up", "x"),
    },
    {
        "name": "attention_naive",
        "crate": "naive-attention",
        "input_file": "attention_inputs.safetensors",
        "output_file": "attention_naive_output.safetensors",
        "output_tensor": "O",
        "reference_op": "attention",
        "reference_inputs": ("Q", "K", "V"),
    },
]


def run_operation(op: dict) -> bool:
    print(f"\n=== {op['name']} ===")

    input_path = DATA_DIR / op["input_file"]
    if not input_path.exists():
        print(f"FAIL: {input_path} not found — run `python3 bench/generators.py` first")
        return False
    output_path = DATA_DIR / op["output_file"]

    inputs = load_file(input_path)
    ref_args = [inputs[name] for name in op["reference_inputs"]]

    ref_start = time.perf_counter()
    expected = REFERENCE_OPS[op["reference_op"]](*ref_args)
    ref_ms = (time.perf_counter() - ref_start) * 1000

    cmd = ["cargo", "run", "-p", op["crate"]]
    if "bin" in op:
        cmd += ["--bin", op["bin"]]
    cmd += ["--release", "--", "--input", str(input_path), "--output", str(output_path)]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print(f"FAIL: {op['name']} kernel run failed (exit {result.returncode})")
        return False

    match = TIMING_RE.search(result.stdout)
    rust_ms = float(match.group(1)) if match else float("nan")

    outputs = load_file(output_path)
    actual = outputs[op.get("output_tensor", "y")]

    report = diff_report(actual, expected)
    tolerance = 1e-3
    passed = report["max_abs_error"] < tolerance

    print(f"max_abs_error       = {report['max_abs_error']:.3e}")
    print(f"mean_abs_error      = {report['mean_abs_error']:.3e}")
    print(f"max_relative_error  = {report['max_relative_error']:.3e}")
    print(f"numpy time          = {ref_ms:.4f} ms")
    print(f"rust kernel time    = {rust_ms:.4f} ms  (avg over 20 dispatches, release build)")
    print(f"{'PASS' if passed else 'FAIL'} (max_abs_error {'<' if passed else '>='} {tolerance:.0e})")

    return passed


def main() -> None:
    results = {op["name"]: run_operation(op) for op in OPERATIONS}

    print("\n=== summary ===")
    for name, passed in results.items():
        print(f"{name:15s} {'PASS' if passed else 'FAIL'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
