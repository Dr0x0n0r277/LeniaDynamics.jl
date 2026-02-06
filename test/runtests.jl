using Test
using LeniaDynamics
using Statistics

@testset "Kernel construction" begin
    spec = KernelSpec(radius=9, rings=Float32[0.5], ring_widths=Float32[0.2], ring_weights=Float32[1.0])
    K = make_kernel(spec, 64)
    @test size(K) == (64,64)
    @test isapprox(sum(K), 1.0f0; atol=1e-4)
    @test minimum(K) >= 0f0
end

@testset "Step invariants" begin
    spec = KernelSpec(radius=7, rings=Float32[0.45,0.8], ring_widths=Float32[0.15,0.1], ring_weights=Float32[1.0,0.6])
    p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.02f0, dt=0.1f0)

    st = initialize_state(64; init=:noise, seed=1, backend=:fft)
    step!(st, p)
    @test all(x -> 0f0 <= x <= 1f0, st.A)
    @test !isnan(mean(st.A))
end

@testset "FFT vs naive consistency (small grid)" begin
    spec = KernelSpec(radius=5, rings=Float32[0.5], ring_widths=Float32[0.2], ring_weights=Float32[1.0])
    p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.02f0, dt=0.1f0)

    st1 = initialize_state(64; init=:spot, seed=7, backend=:fft)
    st2 = initialize_state(64; init=:spot, seed=7, backend=:naive)

    step!(st1, p)
    step!(st2, p)

    err = mean(abs.(st1.A .- st2.A))
    @test err < 2e-2
end


@testset "Integrator sanity" begin
    spec = KernelSpec(radius=6, rings=Float32[0.5], ring_widths=Float32[0.2], ring_weights=Float32[1.0])
    p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.02f0, dt=0.08f0)

    st_e  = initialize_state(64; init=:spot, seed=3, backend=:fft)
    st_r2 = initialize_state(64; init=:spot, seed=3, backend=:fft)
    st_r4 = initialize_state(64; init=:spot, seed=3, backend=:fft)

    step!(st_e,  p; integrator=Euler())
    step!(st_r2, p; integrator=RK2())
    step!(st_r4, p; integrator=RK4())

    @test all(x -> 0f0 <= x <= 1f0, st_e.A)
    @test all(x -> 0f0 <= x <= 1f0, st_r2.A)
    @test all(x -> 0f0 <= x <= 1f0, st_r4.A)
end


@testset "Auto calibration improves mean(U)" begin
    spec = KernelSpec(radius=9, rings=Float32[0.45,0.75], ring_widths=Float32[0.15,0.12], ring_weights=Float32[1.0,0.7])
    p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.015f0, dt=0.10f0)
    st = initialize_state(64; init=:noise, seed=11, backend=:fft)
    st.cache[:params] = p
    U = similar(st.A)
    LeniaDynamics._convolve_fft!(U, st, p, st.A)
    v1 = mean(U)
    auto_calibrate!(st, p; target=p.μ)
    st.cache[:params] = p
    LeniaDynamics._convolve_fft!(U, st, p, st.A)
    v2 = mean(U)
    @test abs(v2 - p.μ) < abs(v1 - p.μ)
end

@testset "Sustain preset stays alive" begin
    st, p, integ = make_preset(:sustain; N=96, seed=1, backend=:fft)
    for _ in 1:250
        step!(st, p; integrator=integ)
    end
    @test mean(st.A) > 0.02
    @test maximum(st.A) > 0.05
end


@testset "Kernel cache invalidation (in-place mutation)" begin
    spec = KernelSpec(radius=6, rings=Float32[0.5], ring_widths=Float32[0.2], ring_weights=Float32[1.0])
    p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.02f0, dt=0.1f0)
    st = initialize_state(64; init=:spot, seed=2, backend=:fft)
    U = similar(st.A)

    LeniaDynamics._convolve_fft!(U, st, p, st.A)
    oldKhat = st.cache[:K̂]
    @test st.cache[:kernel_spec] !== p.kernel  # cached spec must not alias live spec

    # Mutate KernelSpec in-place: should force rebuild of kernel spectrum.
    spec.ring_weights[1] += 0.25f0
    LeniaDynamics._convolve_fft!(U, st, p, st.A)
    @test st.cache[:K̂] !== oldKhat
end

@testset "CUDA backend normalization and round-trip (optional)" begin
    if LeniaDynamics.has_cuda()
        @eval using CUDA
        if CUDA.functional()
            spec = KernelSpec(radius=5, rings=Float32[0.5], ring_widths=Float32[0.2], ring_weights=Float32[1.0])
            p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.02f0, dt=0.1f0)

            st = initialize_state(64; init=:spot, seed=7, backend=:fft)
            Ucpu = zeros(Float32, 64, 64)
            LeniaDynamics._convolve_fft!(Ucpu, st, p, st.A)

            stg = to_device(st, :cuda)
            stg.cache[:params] = p
            Ug = CUDA.zeros(Float32, 64, 64)
            LeniaDynamics._convolve_cuda!(Ug, stg, stg.A)
            Ug_cpu = Array(Ug)

            @test mean(abs.(Ug_cpu .- Ucpu)) < 5e-4

            # Round-trip: cuda -> cpu must copy data.
            st_back = to_device(stg, :fft)
            @test st_back.backend == :fft
            @test st_back.A isa Matrix{Float32}
            @test mean(abs.(st_back.A .- Array(stg.A))) < 1e-6
        end
    end
end
