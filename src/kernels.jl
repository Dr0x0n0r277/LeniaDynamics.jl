# Kernel construction and growth functions.

"""
    gaussian_growth(u, μ, σ) -> g

Smooth growth function often used in Lenia-like systems:
g = 2 * exp(-((u-μ)^2)/(2σ^2)) - 1
"""
@inline function gaussian_growth(u::Real, μ::Real, σ::Real)
    x = (u - μ) / σ
    return 2 * exp(-0.5 * x * x) - 1
end

"""
    bump_growth(u, μ, σ) -> g

Smooth bump-like growth (steeper shoulders).
"""
@inline function bump_growth(u::Real, μ::Real, σ::Real)
    x = abs(u - μ) / σ
    # Smooth bump: exp(-x^4) mapped to [-1, 1]
    return 2 * exp(-(x^4)) - 1
end

@inline function _ring_profile(r::Float32, c::Float32, w::Float32)
    x = (r - c) / w
    return exp(-0.5f0 * x * x)
end

"""
    make_kernel(spec::KernelSpec, N::Int) -> K::Matrix{Float32}

Create an NxN periodic convolution kernel centered for FFT-based circular convolution.
Kernel is radially symmetric and normalized to sum(K) == 1 (within float error).

Implementation detail:
For periodic convolution via FFT, we place the kernel's "center" at (1,1) in FFT layout.
"""

function make_kernel(spec::KernelSpec, N::Int)
    @assert N > 2*spec.radius "Grid N must be larger than 2*radius"
    K = zeros(Float32, N, N)

    rmax = spec.radius
    rmaxf = Float32(rmax)
    rmax2 = rmaxf * rmaxf

    rings = spec.rings
    widths = spec.ring_widths
    weights = spec.ring_weights
    @assert length(rings) == length(widths) == length(weights)

    # Only fill the non-zero support: O(radius^2) instead of O(N^2).
    @inbounds for dx in -rmax:rmax
        ix = mod1(1 + dx, N)  # FFT layout center at (1,1)
        dx2 = Float32(dx*dx)
        for dy in -rmax:rmax
            r2 = dx2 + Float32(dy*dy)
            if r2 <= rmax2
                r = sqrt(r2)
                rr = r / rmaxf
                v = 0f0
                for k in eachindex(rings)
                    v += Float32(weights[k]) * _ring_profile(Float32(rr), Float32(rings[k]), Float32(widths[k]))
                end
                iy = mod1(1 + dy, N)
                K[ix, iy] = v
            end
        end
    end

    s = sum(K)
    @assert s > 0f0 "Kernel sum is zero; check KernelSpec"
    K ./= s
    return K
end
