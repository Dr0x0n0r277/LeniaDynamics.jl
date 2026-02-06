# FFT backend (CPU): caches FFT plans, kernel spectrum, and temporary buffers.

using FFTW
using LinearAlgebra

"""
    _prepare_fft_cache!(st, p)

Create or update FFT caches in `st.cache` for current grid size and kernel spec.

Caches:
- `:N`          : grid size
- `:kernel_spec`: last kernel spec used
- `:plan_f`     : forward FFT plan (out-of-place)
- `:plan_b`     : inverse FFT plan (out-of-place, normalized like `ifft`)
- `:K̂`         : kernel spectrum (ComplexF32)
- `:Ac`         : complex buffer for input (A)
- `:Â`         : complex buffer for spectrum(A)
- `:Uc`         : complex buffer for inverse FFT result
"""
function _prepare_fft_cache!(st::LeniaState{Float32}, p::LeniaParams)
    N1, N2 = size(st.A)
    @assert N1 == N2 "Only square grids are supported"
    N = N1

    if get(st.cache, :N, nothing) == N && get(st.cache, :kernel_spec, nothing) == p.kernel
        return st
    end

    # Build kernel in spatial domain and transform to frequency domain.
    K = make_kernel(p.kernel, N)

    # Allocate buffers (ComplexF32).
    Ac  = Matrix{ComplexF32}(undef, N, N)
    Â   = Matrix{ComplexF32}(undef, N, N)
    Uc  = Matrix{ComplexF32}(undef, N, N)
    Kc  = Matrix{ComplexF32}(undef, N, N)
    K̂   = Matrix{ComplexF32}(undef, N, N)

    @inbounds @simd for i in eachindex(K)
        Kc[i] = ComplexF32(K[i], 0f0)
    end

    # Use ESTIMATE to avoid long planning times on Windows.
    plan_f = plan_fft(Ac; flags=FFTW.ESTIMATE)
    plan_b = plan_ifft(Ac; flags=FFTW.ESTIMATE)

    mul!(K̂, plan_f, Kc)

    st.cache[:N] = N
    st.cache[:kernel_spec] = deepcopy(p.kernel)
    st.cache[:plan_f] = plan_f
    st.cache[:plan_b] = plan_b
    st.cache[:K̂] = K̂
    st.cache[:Ac] = Ac
    st.cache[:Â] = Â
    st.cache[:Uc] = Uc
    return st
end

"""
    _convolve_fft!(U, st, p, A)

Compute `U = K * A` using circular convolution via FFT (CPU).
All arrays are Float32 matrices; internal buffers are ComplexF32.

This is allocation-free once cache is prepared.
"""
function _convolve_fft!(U::AbstractMatrix{Float32}, st::LeniaState{Float32}, p::LeniaParams, A::AbstractMatrix{Float32})
    _prepare_fft_cache!(st, p)

    plan_f = st.cache[:plan_f]
    plan_b = st.cache[:plan_b]
    K̂ = st.cache[:K̂]
    Ac = st.cache[:Ac]
    Â  = st.cache[:Â]
    Uc = st.cache[:Uc]

    @inbounds @simd for i in eachindex(A)
        Ac[i] = ComplexF32(A[i], 0f0)
    end

    mul!(Â, plan_f, Ac)

    @inbounds @simd for i in eachindex(Â)
        Â[i] = Â[i] * K̂[i]
    end

    mul!(Uc, plan_b, Â)

    @inbounds @simd for i in eachindex(U)
        U[i] = real(Uc[i])
    end

    return U
end


"""
    _convolve_fft!(U, st, A)

Backward-compatible helper that uses `st.cache[:params]` as `LeniaParams`.
This is *not* the preferred API (use the 4-arg method), but it can be convenient
for exploratory REPL usage.

Throws `ArgumentError` if `:params` is not present in `st.cache`.
"""
function _convolve_fft!(U::AbstractMatrix{Float32}, st::LeniaState{Float32}, A::AbstractMatrix{Float32})
    p = get(st.cache, :params, nothing)
    p === nothing && throw(ArgumentError("st.cache[:params] missing. Call _convolve_fft!(U, st, p, A) or set st.cache[:params] = p."))
    return _convolve_fft!(U, st, p, A)
end
