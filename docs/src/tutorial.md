# Tutorial: Conway → Lenia

Conway's Game of Life evolves a **binary** grid with a **local, discrete** rule.
**Lenia** replaces this with a **continuous field** and a **smooth, non-local** update.

This page explains the mapping and shows how to run LeniaDynamics.jl.

## 1. State: from bits to a continuous field

- **Conway:** cell state is `0/1`.
- **Lenia:** the state is a real-valued field `A(x, y) ∈ [0, 1]`.

In LeniaDynamics.jl this field lives in `st.A`.

## 2. Neighborhood: from integer counts to convolution

In Conway you count the alive neighbors in a 3×3 neighborhood.
In Lenia you compute a **potential field** `U` by convolving the current state `A` with a kernel `K`:

`U = K ∗ A`

LeniaDynamics supports multiple backends:

- `backend=:fft` uses FFT-based circular convolution (fast for medium/large grids)
- `backend=:naive` uses direct convolution (threaded; useful for validation and tiny kernels)
- `backend=:cuda` is optional if CUDA.jl is installed

## 3. Growth: smooth birth/death instead of thresholds

Conway uses integer thresholds (under/overpopulation).
Lenia uses a **smooth growth function** `g(U; μ, σ)` that returns values roughly in `[-1, 1]`.
A common choice is a Gaussian-shaped growth centered at `μ` with width `σ`.

The field is then updated by integrating

`dA/dt = g(U)`

and clamping back to `[0, 1]`.

## 4. Time integration

Lenia is often presented with explicit Euler integration:

`Aₙ₊₁ = clamp(Aₙ + dt * g(Uₙ), 0, 1)`

LeniaDynamics also includes explicit RK2/RK4 (useful for stability experiments).

## 5. Minimal working example

```julia
using LeniaDynamics

# Kernel: a weighted mixture of rings ("donut" bands)
spec = KernelSpec(
    radius=13,
    rings=Float32[0.45, 0.75],
    ring_widths=Float32[0.15, 0.12],
    ring_weights=Float32[1.0, 0.7],
)

p = LeniaParams(
    kernel=spec,
    growth=gaussian_growth,
    μ=0.15f0,
    σ=0.015f0,
    dt=0.10f0,
)

st = initialize_state(256; init=:spot, seed=42, backend=:fft)
run!(st, p, 200; integrator=RK2())

# Result is in st.A (Matrix{Float32} on CPU backends)
```

## 6. Presets (recommended for demos)

The package ships with presets tuned for stability and long runs:

```julia
st, p, integ = make_preset(:primordia; N=256, seed=42, backend=:fft)
run!(st, p, 500; integrator=integ)
```

If you want a "stays alive" demo, try:

```julia
st, p, integ = make_preset(:sustain; N=256, seed=1, backend=:fft)
run!(st, p, 2000; integrator=integ)
```

## 7. Where to go next

- See the **Performance** page for benchmarking and backend selection.
- Run `scripts/run_gui.jl` for an interactive Makie GUI.
