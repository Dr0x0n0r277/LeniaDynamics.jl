# Time integration and RHS evaluation.
# dA/dt = G(K * A; μ, σ)

"""
    rhs!(dA, st, p, A)

Compute the right-hand side `dA .= G(K*A; μ, σ)` into `dA`.

- `A` may be a CPU array (`Matrix`) or GPU array (`CuArray`) depending on backend.
- `dA` must be allocated on the same device/type as `A`.
"""
function rhs!(dA::AbstractArray{Float32,2}, st::LeniaState{Float32}, p::LeniaParams, A::AbstractArray{Float32,2})
    st.cache[:params] = p
    U = get!(st.cache, :Utmp) do
        similar(A)
    end

    if st.backend == :fft
        _convolve_fft!(U, st, p, A)
    elseif st.backend == :naive
        Ksmall = get!(st.cache, :Ksmall) do
        _make_small_kernel(p.kernel)
    end
        _convolve_naive!(U, A, Ksmall, p.kernel.radius)
    elseif st.backend == :cuda
        _convolve_cuda!(U, st, A)
    else
        error("Unknown backend=$(st.backend).")
    end

    μ, σ = p.μ, p.σ
    growth = p.growth

    # Broadcasted for CPU/GPU. Custom growth must be GPU-compatible for CUDA backend.
    @. dA = Float32(growth(U, μ, σ))
    return dA
end

"""
    integrate_step!(st, p, integrator)

Advance state by one step using the chosen integrator.
Clamps `A` into [0,1] after the step.
"""
function integrate_step!(st::LeniaState{Float32}, p::LeniaParams, ::Euler)
    dt = p.dt
    dA = get!(st.cache, :dA) do
        similar(st.A)
    end
    rhs!(dA, st, p, st.A)
    @. st.A = clamp(st.A + dt*dA, 0f0, 1f0)
    return st
end

function integrate_step!(st::LeniaState{Float32}, p::LeniaParams, ::RK2)
    dt = p.dt
    A0 = st.A
    k1 = get!(st.cache, :k1) do
        similar(A0)
    end
    k2 = get!(st.cache, :k2) do
        similar(A0)
    end
    Atmp = get!(st.cache, :Atmp) do
        similar(A0)
    end

    rhs!(k1, st, p, A0)
    @. Atmp = A0 + (dt/2f0)*k1
    rhs!(k2, st, p, Atmp)
    @. st.A = clamp(A0 + dt*k2, 0f0, 1f0)
    return st
end

function integrate_step!(st::LeniaState{Float32}, p::LeniaParams, ::RK4)
    dt = p.dt
    A0 = st.A
    k1 = get!(st.cache, :k1) do
        similar(A0)
    end
    k2 = get!(st.cache, :k2) do
        similar(A0)
    end
    k3 = get!(st.cache, :k3) do
        similar(A0)
    end
    k4 = get!(st.cache, :k4) do
        similar(A0)
    end
    Atmp = get!(st.cache, :Atmp) do
        similar(A0)
    end

    rhs!(k1, st, p, A0)
    @. Atmp = A0 + (dt/2f0)*k1
    rhs!(k2, st, p, Atmp)
    @. Atmp = A0 + (dt/2f0)*k2
    rhs!(k3, st, p, Atmp)
    @. Atmp = A0 + dt*k3
    rhs!(k4, st, p, Atmp)

    @. st.A = clamp(A0 + (dt/6f0)*(k1 + 2f0*k2 + 2f0*k3 + k4), 0f0, 1f0)
    return st
end
