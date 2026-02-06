
"""
    AbstractFeedback

Marker supertype for optional global feedback controllers that make Lenia dynamics more robust.
"""
abstract type AbstractFeedback end

"""
    MassFeedback(; ρ=0.12f0, κ=0.5f0, mode=:additive, period=1, clamp_scale=(0.5f0, 2.0f0))

A simple global homeostasis controller that prevents "everything dies" (extinction) and
reduces runaway saturation.

The controller measures the current mean density `m = mean(A)` and nudges the state toward
a target `ρ`.

- `mode=:additive`: `A ← clamp(A + dt * κ*(ρ - m), 0, 1)`
- `mode=:rescale` : `A ← clamp(A * s, 0, 1)` where `s = clamp(ρ/max(m,eps), clamp_scale...)`

`period` applies the control only every `period` steps (useful to reduce overhead on GPU).

This is not part’s of "classic" Lenia, but is extremely useful for interactive exploration
and educational demos because it makes simulations far less fragile.
"""
Base.@kwdef struct MassFeedback <: AbstractFeedback
    ρ::Float32 = 0.12f0
    κ::Float32 = 0.5f0
    mode::Symbol = :additive
    period::Int = 1
    clamp_scale::Tuple{Float32,Float32} = (0.5f0, 2.0f0)
end
