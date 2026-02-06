"""
    KernelSpec(; radius, rings, ring_widths, ring_weights)

Radially symmetric kernel specification for Lenia.

- `radius::Int`: kernel radius in cells
- `rings::Vector{Float32}`: ring centers (0..1)
- `ring_widths::Vector{Float32}`: ring widths (0..1)
- `ring_weights::Vector{Float32}`: ring weights (not necessarily normalized)

The resulting kernel is normalized to sum to 1.
"""
Base.@kwdef struct KernelSpec
    radius::Int = 13
    rings::Vector{Float32} = Float32[0.5]
    ring_widths::Vector{Float32} = Float32[0.15]
    ring_weights::Vector{Float32} = Float32[1.0]
end

"""
    LeniaParams(; kernel, growth, μ, σ, dt)

Lenia parameters:
- `kernel::KernelSpec`
- `growth::Function`: growth mapping G(u) -> Δa in `[-1, 1]` (typically)
- `μ, σ`: growth parameters (used by default growth functions)
- `dt`: time step
"""
Base.@kwdef struct LeniaParams
    kernel::KernelSpec = KernelSpec()
    growth::Function = gaussian_growth
    μ::Float32 = 0.15f0
    σ::Float32 = 0.015f0
    dt::Float32 = 0.1f0
    feedback::Union{Nothing,AbstractFeedback} = nothing
end

"""
    LeniaState(A; backend=:fft)

Simulation state containing the grid `A` and backend-specific caches.
"""
mutable struct LeniaState{T, AT<:AbstractArray{T,2}}
    A::AT
    backend::Symbol
    cache::Dict{Symbol, Any}
end

has_cuda() = false  # overwritten if CUDA.jl is available

"""
    set_fftw_threads!(n)

Set FFTW thread count (affects FFT backend on CPU).
"""
function set_fftw_threads!(n::Integer)
    FFTW.set_num_threads(Int(n))
    return n
end


# --- CUDA extension hooks (Julia package extensions) ---
# If CUDA.jl is installed, ext/LeniaDynamicsCUDAExt.jl will load automatically and
# override these methods.

function _to_cuda(st::LeniaState{Float32})
    error("CUDA backend requested but CUDA.jl is not available. Add CUDA.jl to your environment.")
end

function _convolve_cuda!(U::AbstractArray{Float32,2}, st::LeniaState{Float32}, A::AbstractArray{Float32,2})
    error("CUDA backend requested but CUDA.jl is not available. Add CUDA.jl to your environment.")
end


# --- Time integrators -------------------------------------------------------

abstract type Integrator end
struct Euler <: Integrator end
struct RK2   <: Integrator end   # explicit midpoint
struct RK4   <: Integrator end

