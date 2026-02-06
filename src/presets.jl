"""
    preset(name::Symbol; N=256)

Return a named preset as a `NamedTuple` with fields:

- `kernel::KernelSpec`
- `growth` (a scalar growth function, broadcastable)
- `μ::Float32`, `σ::Float32`, `dt::Float32`
- `integrator::Integrator` (Euler/RK2/RK4)
- `init::Symbol` and `init_kwargs::NamedTuple` for `initialize_state`
- `feedback::Union{Nothing,AbstractFeedback}` (optional global homeostasis)
- `autocalibrate::Bool` (scale init so mean(K*A) ≈ μ)

Presets are intended as practical starting points for exploration and GUI defaults.
"""
function preset(name::Symbol; N::Int=256)
    if name == :default
        spec = KernelSpec(radius=13,
                          rings=Float32[0.45, 0.75],
                          ring_widths=Float32[0.15, 0.12],
                          ring_weights=Float32[1.0, 0.7])
        return (kernel=spec, growth=gaussian_growth,
                μ=0.15f0, σ=0.015f0, dt=0.10f0,
                integrator=RK2(),
                init=:spot,
                init_kwargs=(noise_amp=0.05f0,),
                feedback=nothing,
                autocalibrate=true)
    elseif name == :primordia
        # Tuned for fragmented, many-small-structures dynamics (visually closer to "primordia-like" fields).
        spec = KernelSpec(radius=9,
                          rings=Float32[0.25, 0.52, 0.80],
                          ring_widths=Float32[0.08, 0.10, 0.12],
                          ring_weights=Float32[1.0, 0.65, 0.45])
        return (kernel=spec, growth=bump_growth,
                μ=0.09f0, σ=0.055f0, dt=0.045f0,
                integrator=RK2(),
                init=:primordia,
                init_kwargs=(noise_amp=0.05f0, sprinkle_sigma=1.8f0, sprinkle_amp=0.75f0, sprinkle_radius=6),
                feedback=MassFeedback(ρ=0.12f0, κ=0.9f0, mode=:additive, period=2),
                autocalibrate=true)
    
    elseif name == :sustain
        # A robust, self-sustaining preset meant for GUI demos: it resists extinction and runaway saturation.
        spec = KernelSpec(radius=11,
                          rings=Float32[0.30, 0.60, 0.85],
                          ring_widths=Float32[0.10, 0.12, 0.10],
                          ring_weights=Float32[1.0, 0.7, 0.35])
        return (kernel=spec, growth=bump_growth,
                μ=0.10f0, σ=0.060f0, dt=0.045f0,
                integrator=RK2(),
                init=:sprinkle,
                init_kwargs=(noise_amp=0.04f0, sprinkle_sigma=2.0f0, sprinkle_amp=0.85f0, sprinkle_radius=7),
                feedback=MassFeedback(ρ=0.12f0, κ=0.8f0, mode=:additive, period=2),
                autocalibrate=true)
elseif name == :noisy
        spec = KernelSpec(radius=11,
                          rings=Float32[0.35, 0.70],
                          ring_widths=Float32[0.10, 0.12],
                          ring_weights=Float32[1.0, 0.8])
        return (kernel=spec, growth=bump_growth,
                μ=0.12f0, σ=0.060f0, dt=0.06f0,
                integrator=Euler(),
                init=:noise,
                init_kwargs=(noise_amp=0.22f0,),
                feedback=MassFeedback(ρ=0.10f0, κ=0.6f0, mode=:additive, period=1),
                autocalibrate=true)
    else
        error("Unknown preset name=$name. Try :default, :primordia, :sustain, :noisy.")
    end
end

"""
    make_preset(name; N=256, seed=1, backend=:fft)

Convenience: create `(st, p, integrator)` from a preset.
"""
function make_preset(name::Symbol; N::Int=256, seed::Int=1, backend::Symbol=:fft)
    pr = preset(name; N=N)
    st = initialize_state(N; init=pr.init, seed=seed, backend=backend, pr.init_kwargs...)
    p  = LeniaParams(kernel=pr.kernel, growth=pr.growth, μ=pr.μ, σ=pr.σ, dt=pr.dt, feedback=pr.feedback)
    if (haskey(pr, :autocalibrate) && pr.autocalibrate)
        auto_calibrate!(st, p; target=p.μ)
    end
    return st, p, pr.integrator
end
