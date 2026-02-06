
# Feedback controllers and robustness helpers.

"""
    apply_feedback!(st, fb; dt=st.cache[:params].dt)

Apply a feedback controller `fb` to the state `st`.

Currently implemented controllers:
- [`MassFeedback`](@ref)

The controller is applied in-place and is compatible with CPU arrays and (when CUDA.jl is present) CuArrays.
"""
function apply_feedback!(st::LeniaState{Float32}, fb::AbstractFeedback; dt::Float32=1f0)
    if fb isa MassFeedback
        return _apply_mass_feedback!(st, fb; dt=dt)
    else
        error("Unknown feedback controller type: $(typeof(fb))")
    end
end

# internal: apply mass feedback with an internal step counter
function _apply_mass_feedback!(st::LeniaState{Float32}, fb::MassFeedback; dt::Float32)
    period = max(fb.period, 1)
    c = get(st.cache, :_fb_counter, 0) + 1
    st.cache[:_fb_counter] = c
    if (c % period) != 0
        return st
    end

    A = st.A
    m = Float32(mean(A))  # works on CPU and (if available) GPU
    ρ = fb.ρ
    κ = fb.κ

    if fb.mode == :additive
        b = κ * (ρ - m)
        @. A = clamp(A + dt*b, 0f0, 1f0)
    elseif fb.mode == :rescale
        lo, hi = fb.clamp_scale
        s = ρ / max(m, 1f-6)
        s = clamp(Float32(s), lo, hi)
        @. A = clamp(A * s, 0f0, 1f0)
    else
        error("MassFeedback: unsupported mode=$(fb.mode). Use :additive or :rescale.")
    end

    return st
end


"""
    auto_calibrate!(st, p; target=p.μ, statistic=:meanU, clamp_scale=(0.25f0, 4f0))

Make initialization *much less fragile* by scaling the initial state so that the kernel response `U = K*A`
is near the growth center `μ`.

This is a practical robustness trick for interactive exploration:

- compute `U = K*A`
- compute `s = target / statistic(U)`
- set `A ← clamp(A*s, 0, 1)`

`statistic=:meanU` uses `mean(U)` (recommended). You can also use `:medianU` for more robustness.

Returns `(scale=s, value=v)` where `v` is the measured statistic of `U` before scaling.
"""
function auto_calibrate!(st::LeniaState{Float32}, p::LeniaParams;
                         target::Float32=p.μ,
                         statistic::Symbol=:meanU,
                         clamp_scale::Tuple{Float32,Float32}=(0.25f0, 4f0))

    # compute U on current backend
    U = get!(st.cache, :Ucalib) do
        similar(st.A)
    end

    if st.backend == :fft
        _convolve_fft!(U, st, p, st.A)
    elseif st.backend == :naive
        Ksmall = get!(st.cache, :Ksmall) do
            _make_small_kernel(p.kernel)
        end
        _convolve_naive!(U, st.A, Ksmall, p.kernel.radius)
    elseif st.backend == :cuda
        _convolve_cuda!(U, st, st.A)
    else
        error("Unknown backend=$(st.backend).")
    end

    v = if statistic == :meanU
        Float32(mean(U))
    elseif statistic == :medianU
        # median may allocate on GPU; this is mostly for CPU use
        Float32(median(vec(Array(U))))
    else
        error("Unknown statistic=$statistic. Use :meanU or :medianU.")
    end

    lo, hi = clamp_scale
    s = target / max(v, 1f-6)
    s = clamp(Float32(s), lo, hi)

    @. st.A = clamp(st.A * s, 0f0, 1f0)
    st.cache[:_fb_counter] = 0  # reset feedback counter after calibration
    return (scale=s, value=v)
end
