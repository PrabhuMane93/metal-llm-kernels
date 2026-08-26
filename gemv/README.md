# gemv

A naive, single-precision GEMV (matrix × vector) kernel, written in Metal Shading Language, dispatched from a Rust host via `objc2-metal`.

GEMV is the most-executed operation in LLM decode — every layer, every token, one matrix × vector multiply. It's memory-bandwidth-bound rather than compute-bound: the kernel spends its time waiting on weight reads from GPU memory, not on the arithmetic itself. That property is exactly why this is the first kernel in the curriculum, and why the eventual fused-quantization version (`../fused-gemv-gemm/`) matters — cutting bytes moved is the actual lever for a memory-bound kernel, not cutting FLOPs.

`src/gemv.metal` holds the kernel — one thread per output row, a plain accumulation loop across the row. `src/main.rs` is the host side: builds the six core Metal objects (device, pipeline, buffers, queue, command buffer, encoder), wires `W`/`x`/`y`/`numCols` into the kernel's argument slots, dispatches, and reads the result back.

## Running it

Run these from the **repo root** (`metal-llm-kernels/`), with the Python venv from the root README already created and activated, and `requirements.txt` already installed.

Input generation and correctness/performance verification live in the shared harness at [`../bench/`](../bench/), not in this crate — `bench/` is the single source of truth for both the generation logic and the NumPy reference, so there's no risk of it drifting out of sync with a second copy living here. The tradeoff: this kernel isn't standalone-runnable-and-verifiable in isolation anymore — even a one-off sanity check goes through `bench/`.

**1. Generate the inputs** (once, or whenever they need regenerating):
```bash
python3 bench/generators.py
```
Writes `bench/data/gemv_inputs.safetensors` — `W` (4096×4096) and `x` (4096,), seeded, reproducible. Shared with `../tiled_gemv/`, since both compute the same `y = W @ x`.

**2. Run + verify, via the harness:**
```bash
python3 bench/run_bench.py
```
Runs `gemv_naive` (this crate) and `gemv_tiled` against the NumPy reference, reporting max/mean/relative error and timing for each.

**3. Run just this binary directly**, without the harness, if you only want the kernel's own output:
```bash
cargo run -p gemv --release -- --input bench/data/gemv_inputs.safetensors --output bench/data/gemv_naive_output.safetensors
```
Prints three sample `y[...]` values and a `TIMING_MS:` line (average over 20 dispatches, 1 warmup excluded). The full result is written to `--output` as a `.safetensors` file, not printed — comparing it against a reference is `bench/compare.py`'s job now, invoked through `run_bench.py`.

## Known simplifications

- **Weight loading reads the whole file into memory** (`std::fs::read`) before copying into the Metal buffer. Fine at this file's size (~64MB); a real loader would `mmap` the file instead (lazy paging, no full up-front read) and could go further with Metal's no-copy buffer creation to avoid the second copy entirely. Worth revisiting once real multi-GB model weights are involved, not for a single test matrix.
- **FP32 only, no quantization yet.** That's the explicit job of `../fused-gemv-gemm/`, the next kernel in the roadmap.
