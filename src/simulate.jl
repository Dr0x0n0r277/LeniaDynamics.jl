# High-level simulation interface.

"""
    initialize_state(N; init=:noise, seed=1, backend=:fft, kwargs...) -> LeniaState

Create an `N×N` state.

Built-in initializers:

- `init=:noise`     – uniform noise in `[0, noise_amp]` (default `noise_amp=0.1`)
- `init=:spot`      – centered Gaussian blob + small noise
- `init=:sprinkle`  – many small Gaussian blobs randomly placed (good for "many small organisms" regimes)
- `init=:primordia` – alias for `:sprinkle` with parameters tuned for fragmented dynamics

Keywords:

- `noise_amp::Float32=0.1f0`
- `spot_sigma_frac::Float32=0.12f0`  (sigma = `spot_sigma_frac*N`)
- `spot_amp::Float32=0.8f0`
- `sprinkle_n::Int` (default scales with N)
- `sprinkle_sigma::Float32=2.0f0`
- `sprinkle_amp::Float32=0.8f0`
- `sprinkle_radius::Int=6` (cutoff radius for blob stamping)

Notes:
- `backend` is stored in the returned state; you can later call `to_device(st, :cuda)` when CUDA is available.
"""
function initialize_state(N::Integer; init::Symbol=:noise, seed::Integer=1, backend::Symbol=:fft,
                          noise_amp::Float32=0.1f0,
                          spot_sigma_frac::Float32=0.12f0,
                          spot_amp::Float32=0.8f0,
                          sprinkle_n::Int=0,
                          sprinkle_sigma::Float32=2.0f0,
                          sprinkle_amp::Float32=0.8f0,
                          sprinkle_radius::Int=6)

    rng = MersenneTwister(seed)
    A = zeros(Float32, N, N)

    # Helper: stamp a small Gaussian blob with periodic boundary conditions.
    function _stamp_blob!(A::AbstractMatrix{Float32}, cx::Int, cy::Int, σ::Float32, amp::Float32, rad::Int)
        inv2σ2 = 1f0 / (2f0*σ*σ)
        @inbounds for dx in -rad:rad, dy in -rad:rad
            x = mod1(cx + dx, N)
            y = mod1(cy + dy, N)
            r2 = Float32(dx*dx + dy*dy)
            A[x,y] += amp * exp(-r2 * inv2σ2)
        end
        return A
    end

    if init == :noise
        A .= noise_amp .* rand(rng, Float32, N, N)

    elseif init == :spot
        cx, cy = (N+1) ÷ 2, (N+1) ÷ 2
        σ = max(1f0, spot_sigma_frac*Float32(N))
        rad = max(6, Int(ceil(3f0*σ)))
        _stamp_blob!(A, cx, cy, σ, spot_amp, rad)
        A .+= (0.5f0*noise_amp) .* rand(rng, Float32, N, N)
        A .= clamp.(A, 0f0, 1f0)

    elseif init == :sprinkle || init == :primordia
        # Defaults that scale with area.
        # For N=256 => about 180 seeds (works well for "fragmented" regimes).
        nseeds = sprinkle_n == 0 ? max(80, Int(round(0.0027 * N * N / 1.0))) : sprinkle_n
        σ = sprinkle_sigma
        amp = sprinkle_amp
        rad = sprinkle_radius

        # "primordia" default tweaks (more fragmented + lighter background)
        local_noise = noise_amp
        if init == :primordia
            local_noise = 0.04f0
            σ = sprinkle_sigma == 2.0f0 ? 1.6f0 : σ
            amp = sprinkle_amp == 0.8f0 ? 0.65f0 : amp
            rad = sprinkle_radius == 6 ? 5 : rad
        end

        @inbounds for _ in 1:nseeds
            cx = rand(rng, 1:N)
            cy = rand(rng, 1:N)
            _stamp_blob!(A, cx, cy, σ, amp, rad)
        end
        A .+= local_noise .* rand(rng, Float32, N, N)
        A .= clamp.(A, 0f0, 1f0)

    else
        error("Unknown init=$init")
    end

    st = LeniaState{Float32, typeof(A)}(A, backend, Dict{Symbol, Any}())
    return st
end

"""
    step!(st, p; method=:auto) -> st

Advance the state by one step using the selected backend:
- `:fft` – CPU FFT convolution (fast)
- `:naive` – CPU direct convolution (threaded; validation)
- `:cuda` – optional CUDA convolution (requires CUDA.jl; see `cuda_optional.jl`)
"""
function step!(st::LeniaState{Float32}, p::LeniaParams; integrator::Integrator=Euler())
    integrate_step!(st, p, integrator)
    if p.feedback !== nothing
        apply_feedback!(st, p.feedback; dt=p.dt)
    end
    return st
end


"""
    run!(st, p, steps; callback=nothing)

Run `steps` iterations. If `callback` is provided, it is called as `callback(step, st)`.
"""
function run!(st::LeniaState{Float32}, p::LeniaParams, steps::Integer; integrator::Integrator=Euler(), callback=nothing)
    for t in 1:steps
        step!(st, p; integrator=integrator)
        if callback !== nothing
            callback(t, st)
        end
    end
    return st
end

# -- GPU/CPU interop helpers (do not hard-depend on CUDA in the core package) --
# If the CUDA extension is loaded, it provides `to_cpu_if_gpu(A) -> (A_cpu, converted::Bool)`.
@inline function _to_cpu_if_gpu(A)
    ext = Base.get_extension(@__MODULE__, :LeniaDynamicsCUDAExt)
    if ext === nothing
        return (A, false)
    end
    return ext.to_cpu_if_gpu(A)
end

"""
    to_device(st, backend::Symbol) -> LeniaState

Switch backend.
- `backend == :cuda` requires CUDA.jl.
- CPU backends: `:fft`, `:naive`.
"""

function to_device(st::LeniaState{Float32}, backend::Symbol)
    if backend == :cuda
        if !has_cuda()
            error("CUDA backend requested but CUDA.jl is not available in this environment.")
        end
        return _to_cuda(st)
    elseif backend in (:fft, :naive)
        # If we're currently holding GPU data, return a fresh CPU state with copied data.
        A_cpu, converted = _to_cpu_if_gpu(st.A)
        if converted
            return LeniaState{Float32, typeof(A_cpu)}(A_cpu, backend, Dict{Symbol, Any}())
        end
        st.backend = backend
        empty!(st.cache)
        return st
    else
        error("Unknown backend=$backend")
    end
end
