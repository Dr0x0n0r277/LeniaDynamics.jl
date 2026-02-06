# Performance

Lenia spends most time in the **convolution** step (computing `U = K ∗ A`).
LeniaDynamics provides multiple implementations so you can trade off correctness checks, speed, and hardware.

## Backends in practice

- `:fft` (CPU, FFTW): best for medium/large grids (typical Lenia sizes 128–1024).
- `:naive` (CPU): simple direct convolution; useful for debugging and small kernels.
- `:cuda` (GPU): optional; requires CUDA.jl and a functional CUDA installation.

## Benchmark script

The repository includes a reproducible benchmark script:

```bash
julia --project=scripts scripts/bench.jl
```

You can control the grid size and steps via environment variables:

```bash
# Linux/macOS
LENIA_N=512 LENIA_STEPS=50 julia --project=scripts scripts/bench.jl

# Windows (PowerShell)
$env:LENIA_N=512; $env:LENIA_STEPS=50; julia --project=scripts scripts/bench.jl
```

The script reports the time per step for `:fft` and `:naive` backends, and `:cuda` if available.

## FFTW threading

FFTW can use multiple CPU threads.
LeniaDynamics exposes a small helper:

```julia
using LeniaDynamics
LeniaDynamics.set_fftw_threads!(Threads.nthreads())
```

A good practice is to benchmark several thread counts (e.g. 1, 2, 4, …) for your target grid size.

## Practical tuning tips

- **Use `:fft`** for most real runs.
- For correctness/regression tests, compare `:fft` vs `:naive` on smaller grids.
- Keep `A` in `Float32` (default) for speed.
- Avoid allocating in the step loop; LeniaDynamics caches FFT plans and kernel spectra per state.

## CUDA notes

GPU acceleration is optional and enabled via the package extension. Install CUDA.jl:

```julia
import Pkg
Pkg.add("CUDA")
```

Then select `backend=:cuda` via `initialize_state(...; backend=:cuda)` or convert an existing state:

```julia
stg = to_device(st, :cuda)
```

In the Makie GUI, rendering every *k* frames can reduce GPU→CPU transfer overhead.
