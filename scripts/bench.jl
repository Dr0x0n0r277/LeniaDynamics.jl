# Benchmark LeniaDynamics backends.
#
# Usage:
#   julia --project=scripts scripts/bench.jl
#   LENIA_N=512 LENIA_STEPS=1 julia --project=scripts scripts/bench.jl

using Pkg
try
    using LeniaDynamics
catch
    Pkg.develop(path=joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    using LeniaDynamics
end

using BenchmarkTools
using Statistics
using FFTW

# Env-controlled parameters
N = parse(Int, get(ENV, "LENIA_N", "0"))
Ns = N > 0 ? [N] : [128, 256, 512]
steps_warmup = parse(Int, get(ENV, "LENIA_WARMUP", "20"))

# A representative Lenia kernel + growth parameters
spec = KernelSpec(
    radius=13,
    rings=Float32[0.45, 0.75],
    ring_widths=Float32[0.15, 0.12],
    ring_weights=Float32[1.0, 0.7],
)

p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.015f0, dt=0.10f0)

println("LeniaDynamics backend benchmark")
println("  Julia threads:  ", Threads.nthreads())
set_fftw_threads!(Threads.nthreads())
println("  FFTW threads:   ", FFTW.get_num_threads())
println("  CUDA available: ", has_cuda())

function warmup!(st, p; n=20)
    for _ in 1:n
        step!(st, p; integrator=Euler())
    end
    return st
end

for N in Ns
    println("\n=== Grid size N = $N ===")

    st_fft = initialize_state(N; init=:spot, seed=42, backend=:fft)
    warmup!(st_fft, p; n=steps_warmup)
    println("FFT backend (one step):")
    @btime step!($st_fft, $p; integrator=Euler())

    st_naive = initialize_state(N; init=:spot, seed=42, backend=:naive)
    warmup!(st_naive, p; n=steps_warmup)
    println("Naive backend (one step):")
    @btime step!($st_naive, $p; integrator=Euler())

    if has_cuda()
        println("CUDA backend (one step):")
        stc = to_device(initialize_state(N; init=:spot, seed=42, backend=:fft), :cuda)
        warmup!(stc, p; n=min(steps_warmup, 5))
        @btime step!($stc, $p; integrator=Euler())
    end
end
