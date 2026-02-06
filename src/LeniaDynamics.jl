module LeniaDynamics

using FFTW
using LinearAlgebra
using Random
using Statistics

export LeniaParams, LeniaState, KernelSpec
export gaussian_growth, bump_growth
export make_kernel, initialize_state, step!, run!, rhs!
export Integrator, Euler, RK2, RK4
export preset, make_preset
export set_fftw_threads!
export to_device, has_cuda
export AbstractFeedback, MassFeedback, apply_feedback!, auto_calibrate!


include("feedback_types.jl")
include("types.jl")
include("kernels.jl")
include("backends_fft.jl")
include("backends_naive.jl")
include("integrators.jl")
include("feedback.jl")
include("presets.jl")
include("simulate.jl")

end # module
