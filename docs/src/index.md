# LeniaDynamics.jl

LeniaDynamics.jl is a reusable, performance-oriented **Lenia (continuous Game of Life)** simulator.
It provides CPU FFT convolution, a threaded naive backend for validation, and optional CUDA acceleration.

## Quick start

```julia
using LeniaDynamics

spec = KernelSpec(
    radius=13,
    rings=Float32[0.45, 0.75],
    ring_widths=Float32[0.15, 0.12],
    ring_weights=Float32[1.0, 0.7],
)

p = LeniaParams(kernel=spec, μ=0.15f0, σ=0.015f0, dt=0.10f0)

st = initialize_state(256; init=:spot, seed=42, backend=:fft)
run!(st, p, 200; integrator=RK2())
```

## Where to start

- Read **Tutorial** for the mapping from Conway → Lenia and a minimal example.
- Read **Performance** for benchmarking, backend selection, and tuning tips.

## GUI / visualization

The interactive GUI is in `scripts/run_gui.jl`.
Run it from the package root:

```bash
julia --project=scripts scripts/run_gui.jl
```

## Presets

```julia
st, p, integ = make_preset(:primordia; N=256, seed=42, backend=:fft)
run!(st, p, 500; integrator=integ)
```
