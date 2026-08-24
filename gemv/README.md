# gemv

A naive, single-precision GEMV (matrix × vector) kernel, written in Metal Shading Language, dispatched from a Rust host via `objc2-metal`.

GEMV is the most-executed operation in LLM decode — every layer, every token, one matrix × vector multiply. It's memory-bandwidth-bound rather than compute-bound: the kernel spends its time waiting on weight reads from GPU memory, not on the arithmetic itself. That property is exactly why this is the first kernel in the curriculum, and why the eventual fused-quantization version (`../fused-gemv-gemm/`) matters — cutting bytes moved is the actual lever for a memory-bound kernel, not cutting FLOPs.

`src/gemv.metal` holds the kernel — one thread per output row, a plain accumulation loop across the row. `src/main.rs` is the host side: builds the six core Metal objects (device, pipeline, buffers, queue, command buffer, encoder), wires `W`/`x`/`y`/`numCols` into the kernel's argument slots, dispatches, and reads the result back.

## Running it

Run these from the **repo root** (`metal-llm-kernels/`), with the Python venv from the root README already created and activated, and `requirements.txt` already installed.

**1. Generate the inputs.** `gen_weights.py` creates both `W` (4096×4096) and `x` (4096,) — seeded random values, saved together as one `gemv_inputs.safetensors` file (Hugging Face's safetensors format). This file is gitignored and regenerated on demand, not committed — the seed makes it reproducible.
```bash
python3 gemv/gen_weights.py
```
Expect: `wrote gemv_inputs.safetensors — W (4096, 4096) float32, x (4096,) float32`

**2. Build and run the Rust kernel.** Loads `gemv_inputs.safetensors`, copies `W` and `x` directly into GPU-shared Metal buffers (no intermediate `Vec<f32>` — the buffer's own memory is the only copy of the data that exists on the Rust side), dispatches the kernel, prints three sample outputs.
```bash
cargo run -p gemv
```
Expect: three lines like `y[0] = -0.010970085`.

**3. Verify against an independent reference.** `verify.py` loads the *same* `gemv_inputs.safetensors` file and computes `W @ x` with NumPy — same input bytes, independent implementation, independent language.
```bash
python3 gemv/verify.py
```
Expect the same three `y[...]` values, matching to float32 tolerance (the last digit or two may differ — the kernel accumulates sequentially while NumPy uses a different reduction order internally; this is normal floating-point rounding, not a bug).

**4. Compare performance (planned, not yet implemented).** A Rust-vs-NumPy timing comparator is planned as the next addition to this kernel — this section will be filled in once that script exists, rather than documenting something that isn't there yet.

## Known simplifications

- **Weight loading reads the whole file into memory** (`std::fs::read`) before copying into the Metal buffer. Fine at this file's size (~64MB); a real loader would `mmap` the file instead (lazy paging, no full up-front read) and could go further with Metal's no-copy buffer creation to avoid the second copy entirely. Worth revisiting once real multi-GB model weights are involved, not for a single test matrix.
- **FP32 only, no quantization yet.** That's the explicit job of `../fused-gemv-gemm/`, the next kernel in the roadmap.
