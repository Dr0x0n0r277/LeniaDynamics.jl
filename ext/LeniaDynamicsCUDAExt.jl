module LeniaDynamicsCUDAExt

using LeniaDynamics
import LeniaDynamics: has_cuda, _to_cuda, _convolve_cuda!
using CUDA
using LinearAlgebra

# Tell the main package CUDA is available.
LeniaDynamics.has_cuda() = true


# Utility for the core package: detect & copy GPU arrays back to CPU without a hard CUDA dependency.
to_cpu_if_gpu(A) = (A, false)
to_cpu_if_gpu(A::CUDA.AbstractGPUArray) = (Array(A), true)


function LeniaDynamics._to_cuda(st::LeniaDynamics.LeniaState{Float32})
    A = CUDA.cu(st.A)
    return LeniaDynamics.LeniaState{Float32, typeof(A)}(A, :cuda, Dict{Symbol,Any}())
end

"""
    _prepare_cuda_cache!(st, p)

Prepare GPU caches:
- kernel spectrum `K̂` on GPU
- (optional) CUFFT plans + temporary buffers for allocation-free steps
"""
function _prepare_cuda_cache!(st::LeniaDynamics.LeniaState{Float32}, p::LeniaDynamics.LeniaParams)
    N = size(st.A,1)
    cacheN = get(st.cache, :N, nothing)
    if cacheN === N && get(st.cache, :kernel_spec, nothing) == p.kernel && get(st.cache, :ifft_scale_N, nothing) == N
        return st
    end

    # Kernel spectrum on GPU
    K = LeniaDynamics.make_kernel(p.kernel, N)
    Kd = CUDA.cu(K)
    K̂ = CUDA.fft(Kd)


    # Determine CUDA ifft normalization once per grid size.
    # AbstractFFTs conventions are that ifft(fft(x)) ≈ x. Some backends may require manual scaling,
    # so we detect and cache the correction factor.
    if get(st.cache, :ifft_scale_N, nothing) != N
        Aimp = CUDA.zeros(Float32, N, N)
        CUDA.@allowscalar Aimp[1,1] = 1f0
        B = CUDA.ifft(CUDA.fft(Aimp))
        b11 = Float32(real(CUDA.@allowscalar B[1,1]))
        target = 1f0
        # scale such that b11*scale ≈ 1
        if abs(b11 - target) < 1e-2
            st.cache[:ifft_scale] = 1f0
        elseif abs(b11 - Float32(N*N)) < 0.05f0*Float32(N*N)
            st.cache[:ifft_scale] = 1f0 / Float32(N*N)
        elseif b11 != 0f0
            st.cache[:ifft_scale] = target / b11
        else
            st.cache[:ifft_scale] = 1f0
        end
        st.cache[:ifft_scale_N] = N
    end

    st.cache[:N] = N
    st.cache[:kernel_spec] = deepcopy(p.kernel)
    st.cache[:K̂] = K̂

    # Try to prepare plans + buffers for allocation-free convolution.
    # If this fails (API differences), we fall back to CUDA.fft / CUDA.ifft each step.
    st.cache[:use_plans] = false
    try
        plan_f = plan_fft(st.A)            # AbstractFFTs plan on CuArray
        plan_b = plan_ifft(K̂)
        Â = similar(K̂)
        Uc = similar(K̂)

        # Quick smoke test: mul!(Â, plan_f, A) must work.
        mul!(Â, plan_f, st.A)

        st.cache[:plan_f] = plan_f
        st.cache[:plan_b] = plan_b
        st.cache[:Â] = Â
        st.cache[:Uc] = Uc
        st.cache[:use_plans] = true
    catch
        # keep fallback
    end

    return st
end

"""
    LeniaDynamics._convolve_cuda!(U, st, A)

CUDA convolution backend (full GPU):
- computes `U = K * A` on GPU via FFT
- writes into preallocated `U` on GPU
"""
function LeniaDynamics._convolve_cuda!(U::CUDA.CuArray{Float32,2}, st::LeniaDynamics.LeniaState{Float32}, A::CUDA.CuArray{Float32,2})
    p = st.cache[:params]
    _prepare_cuda_cache!(st, p)
    K̂ = st.cache[:K̂]

    if get(st.cache, :use_plans, false)
        plan_f = st.cache[:plan_f]
        plan_b = st.cache[:plan_b]
        Â = st.cache[:Â]
        Uc = st.cache[:Uc]

        mul!(Â, plan_f, A)
        @. Â = Â * K̂
        mul!(Uc, plan_b, Â)
        scale = st.cache[:ifft_scale]
        @. U = Float32(real(Uc)) * scale
        return U
    else
        # Fallback (allocates temporaries)
        Â = CUDA.fft(A)
        Â .*= K̂
        Uc = CUDA.ifft(Â)
        scale = st.cache[:ifft_scale]
        @. U = Float32(real(Uc)) * scale
        return U
    end
end

end # module LeniaDynamicsCUDAExt
