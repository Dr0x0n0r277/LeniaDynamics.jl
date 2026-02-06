# Naive spatial convolution backend (CPU) with Threads.

using Base.Threads

"""
    _convolve_naive!(U, A, Ksmall; radius)

Compute U = K * A using direct summation with periodic boundaries.
O(N^2 * r^2): intended mainly for validation / small kernels.
"""
function _convolve_naive!(U::AbstractMatrix{Float32}, A::AbstractMatrix{Float32}, Ksmall::AbstractMatrix{Float32}, radius::Int)
    N = size(A,1)
    @assert size(A,1) == size(A,2) == size(U,1) == size(U,2)
    @assert size(Ksmall,1) == size(Ksmall,2) == 2*radius+1

    @threads for i in 1:N
        for j in 1:N
            s = 0f0
            @inbounds for di in -radius:radius
                ii = mod1(i + di, N)
                for dj in -radius:radius
                    jj = mod1(j + dj, N)
                    s += Ksmall[di+radius+1, dj+radius+1] * A[ii, jj]
                end
            end
            U[i,j] = s
        end
    end
    return U
end

"""
    _make_small_kernel(spec::KernelSpec) -> Ksmall

Construct a small (2r+1)x(2r+1) kernel for naive convolution.
"""
function _make_small_kernel(spec::KernelSpec)
    r = spec.radius
    K = zeros(Float32, 2r+1, 2r+1)
    rmax = Float32(r)

    rings = spec.rings
    widths = spec.ring_widths
    weights = spec.ring_weights

    for di in -r:r, dj in -r:r
        rr = sqrt(Float32(di*di + dj*dj))
        if rr <= rmax
            x = rr / rmax
            v = 0f0
            @inbounds for k in eachindex(rings)
                v += Float32(weights[k]) * exp(-0.5f0 * ((Float32(x) - Float32(rings[k])) / Float32(widths[k]))^2)
            end
            K[di+r+1, dj+r+1] = v
        end
    end
    K ./= sum(K)
    return K
end
