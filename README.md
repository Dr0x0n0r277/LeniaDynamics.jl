# LeniaDynamics.jl

A reusable, performance-oriented **Lenia (continuous Game of Life)** simulator in **Julia**, focused on:

- fast evolution via **FFT-based convolution** (CPU / FFTW),
- a threaded **naive** backend for validation and tiny kernels,
- optional **CUDA** acceleration (GPU) if `CUDA.jl` is installed,
- reproducible scripts, benchmarks, tests, and basic docs.

## Quick start (package API)

```julia
using LeniaDynamics

spec = KernelSpec(
    radius=13,
    rings=Float32[0.45, 0.75],
    ring_widths=Float32[0.15, 0.12],
    ring_weights=Float32[1.0, 0.7],
)

p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.015f0, dt=0.10f0)

st = initialize_state(256; init=:spot, seed=42, backend=:fft)
run!(st, p, 200; integrator=RK2())
# result is in st.A
```

## Installation / running tests

From the package root:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

For local development (REPL):

```julia
using Pkg
Pkg.develop(path="/path/to/LeniaDynamics")
Pkg.instantiate()
```

If you host the repository online, you can install via URL:

```julia
using Pkg
Pkg.add(url="https://github.com/USER/REPO.git")
```

## Scripts (reproducible environment)

The `scripts/` folder contains a separate Julia environment (committed `Project.toml` + `Manifest.toml`).

- Example run + microbenchmarks:

```bash
julia --project=scripts scripts/run_example.jl
```

- Backend benchmark sweep:

```bash
julia --project=scripts scripts/bench.jl
```

- GUI (GLMakie):

```bash
julia --project=scripts scripts/run_gui.jl
```

## CUDA (optional)

CUDA support is provided via a package extension. To enable it, add CUDA to **your environment**:

```julia
import Pkg
Pkg.add("CUDA")
```

Then use `backend=:cuda` (or `to_device(st, :cuda)`).

## Project structure

- `src/` — package code
- `test/` — runnable tests (`Pkg.test()`)
- `docs/` — Documenter.jl site (`julia --project=docs docs/make.jl`)
- `scripts/` — reproducible examples, GUI, benchmarks
- `.github/workflows/ci.yml` — GitHub Actions CI

## License

MIT
