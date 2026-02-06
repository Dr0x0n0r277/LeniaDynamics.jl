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

spec = KernelSpec(radius=13,
                  rings=Float32[0.45, 0.75],
                  ring_widths=Float32[0.15, 0.12],
                  ring_weights=Float32[1.0, 0.7])

p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.015f0, dt=0.10f0)

N = 256
st = initialize_state(N; init=:spot, seed=42, backend=:fft)

set_fftw_threads!(Threads.nthreads())

println("Running a short simulation (50 steps)...")
run!(st, p, 50; integrator=RK2())

println("Stats after 50 steps:")
Ahost = (typeof(st.A).name.name == :CuArray) ? Array(st.A) : st.A
println("  mean(A) = ", mean(Ahost))
println("  max(A)  = ", maximum(Ahost))

println("\nBenchmarking one step (FFT backend)...")
@btime step!($st, $p; integrator=Euler())

st_naive = initialize_state(N; init=:spot, seed=42, backend=:naive)
println("\nBenchmarking one step (naive backend, threaded)...")
@btime step!($st_naive, $p; integrator=Euler())

if has_cuda()
    println("\nCUDA detected: benchmarking CUDA convolution (hybrid: convolution on GPU, update on CPU)...")
    stc = to_device(initialize_state(N; init=:spot, seed=42, backend=:fft), :cuda)
    @btime step!($stc, $p; integrator=Euler())
else
    println("\nCUDA not available (optional). To enable:")
    println("  import Pkg; Pkg.add(\"CUDA\")")
end
