mutable struct Hyperparameters{T}
    eps::T
    L::T
    nu::T
    lambda_c::T
    sigma::AbstractVector{T}
    gamma::T
    sigma_xi::T
    Weps::T
    Feps::T
end

function Hyperparameters(; kwargs...)
    eps = get(kwargs, :eps, 0.0)
    L = get(kwargs, :L, 0.0)
    nu = get(kwargs, :nu, 0.0)
    sigma = get(kwargs, :sigma, [0.0])
    lambda_c = get(kwargs, :lambda_c, 0.0)
    gamma = get(kwargs, :gamma, 0.0)
    sigma_xi = get(kwargs, :sigma_xi, 0.0)
    Weps = get(kwargs, :Weps, 0.0)
    Feps = get(kwargs, :Feps, 0.0)
    return Hyperparameters(eps, L, nu, lambda_c, sigma, gamma, sigma_xi, Weps, Feps)
end

mutable struct MCHMCSampler <: AbstractMCMC.AbstractSampler
    nadapt::Int
    TEV::Real
    adaptive::Bool
    tune_eps::Bool
    tune_L::Bool
    tune_sigma::Bool
    hyperparameters::Hyperparameters
    hamiltonian_dynamics::Function
end


"""
    MCHMC(
        nadapt::Int,
        TEV::Real;
        kwargs...
    )
Constructor for the MicroCanonical HMC (q=0 Hamiltonian) sampler
"""
function MCHMC(nadapt::Int, TEV::Real;
    integrator="LF",
    adaptive=false,
    tune_eps=true,
    tune_L=true,
    tune_sigma=true,
    kwargs...)

    ### Init Hyperparameters ###
    hyperparameters = Hyperparameters(;kwargs...)

    ### integrator ###
    if integrator == "LF" # leapfrog
        hamiltonian_dynamics = Leapfrog
    elseif integrator == "MN" # minimal norm
        hamiltonian_dynamics = Minimal_norm
    else
        println(string("integrator = ", integrator, "is not a valid option."))
    end

    return MCHMCSampler(nadapt, TEV, adaptive, tune_eps, tune_L, tune_sigma, hyperparameters, hamiltonian_dynamics)
end

function Random_unit_vector(rng::AbstractRNG, d::Int; _normalize = true)
    return Random_unit_vector(rng, d, Float64; _normalize = _normalize)
end

function Random_unit_vector(rng::AbstractRNG, d::Int, T::Type; _normalize = true)
    """Generates a random (isotropic) unit vector."""
    u = randn(rng, T, d)
    if _normalize
        u = normalize(u)
    end
    return u
end

function Partially_refresh_momentum(rng::AbstractRNG, nu::T, u::Vector{T}) where {T}
    d = length(u)
    z = nu .* Random_unit_vector(rng, d, T; _normalize = false)
    uu = u .+ z
    return normalize(uu)
end

function Update_momentum(d::Int, eff_eps::T, g::Vector{T}, u::Vector{T}) where {T}
    """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
    similar to the implementation: https://github.com/gregversteeg/esh_dynamics
    There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
    g_norm = norm(g)
    e = -g ./ g_norm
    ue = dot(u, e)
    delta = eff_eps * g_norm / (d - 1)
    zeta = exp(-delta)
    uu = e .* ((1 - zeta) * (1 + zeta + ue * (1 - zeta))) + (2 * zeta) .* u
    delta_r = delta - log(2) + log(1 + ue + (1 - ue) * zeta^2)
    return normalize(uu), delta_r
end

struct MCHMCState{T}
    rng::AbstractRNG
    i::Int
    x::Vector{T}
    u::Vector{T}
    l::T
    g::Vector{T}
    dE::T
    h::Hamiltonian
end

struct Transition{T}
    θ::Vector{T}
    ϵ::T
    δE::T
    ℓ::T
end

function Transition(
    state::MCHMCState{T},
    hp::Hyperparameters{T},
    bijector) where {T}
    eps = (hp.Feps / hp.Weps)^(-1 / 6)
    sample = bijector(state.x)[:]
    return Transition(sample, T(eps), state.dE, -state.l)
end

function Step(
    sampler::MCHMCSampler,
    h::Hamiltonian,
    init_params::AbstractVector;
    kwargs...)
    return Step(Random.GLOBAL_RNG, sampler, h, init_params; kwargs...)
end

function Step(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    h::Hamiltonian,
    init_params::Vector{T};
    bijector = NoTransform,
    kwargs...,
) where {T}
    kwargs = Dict(kwargs)
    d = length(init_params)
    l, g = -1 .* h.∂lπ∂θ(init_params)
    u = Random_unit_vector(rng, d, T)
    state = MCHMCState{T}(rng, 0, init_params, u, l, g, T(0.0), h)
    state = tune_hyperparameters(rng, sampler, state; kwargs...)
    transition = Transition(state, sampler.hyperparameters, bijector)
    return transition, state
end

function Step(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    state::MCHMCState{T};
    bijector = NoTransform,
    kwargs...,
) where {T}
    """One step of the Langevin-like dynamics."""
    dialog = get(kwargs, :dialog, false)

    eps = sampler.hyperparameters.eps
    Weps = sampler.hyperparameters.Weps
    Feps = sampler.hyperparameters.Feps
    nu = sampler.hyperparameters.nu
    sigma_xi = sampler.hyperparameters.sigma_xi
    gamma = sampler.hyperparameters.gamma

    TEV = sampler.TEV

    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, state)
    # Langevin-like noise
    uuu = Partially_refresh_momentum(rng, nu, uu)
    dEE = T(kinetic_change + ll - state.l)

    if sampler.adaptive
        d = length(xx)
        varE = dEE^2 / d
        # 1e-8 is added to avoid divergences in log xi        
        xi = varE / TEV + T(1e-8)
        # the weight which reduces the impact of stepsizes which 
        # are much larger on much smaller than the desired one.        
        w = exp(-(1/2) * (log(xi) / (6 * sigma_xi))^2)
        # Kalman update the linear combinations
        new_Feps = gamma * Feps + w * (xi / eps^6)
        new_Weps = gamma * Weps + w
        new_eps = (Feps / Weps)^(-1 / 6)

        sampler.hyperparameters.Feps = T(new_Feps)
        sampler.hyperparameters.Weps = T(new_Weps)
        sampler.hyperparameters.eps = T(new_eps)
        tune_nu!(sampler, d)
    end

    state = MCHMCState(rng, state.i + 1, xx, uuu, ll, gg, dEE, state.h)
    transition = Transition(state, sampler.hyperparameters, bijector)
    return transition, state
end

function Sample(
    sampler::MCHMCSampler,
    target::Target,
    n::Int;
    fol_name = ".",
    file_name = "samples",
    progress = true,
    kwargs...,
)
    return Sample(
        Random.GLOBAL_RNG,
        sampler,
        target,
        n;
        fol_name = fol_name,
        file_name = file_name,
        kwargs...,
    )
end

"""
    sample(
        rng::AbstractRNG,
        sampler::MCHMCSampler,
        target::Target,
        n::Int;
        init_params = nothing,
        fol_name = ".",
        file_name = "samples",
        progress = true,
        kwargs...
    )
Sampling routine
"""

function _make_sample(transition::Transition)
    return [transition.θ; transition.ϵ; transition.δE; transition.ℓ]
end

"""
    $(TYPEDSIGNATURES)

Sample from the target distribution using the provided sampler.

Keyword arguments:
* `file_name` — if provided, save chain to disk (in HDF5 format)
* `file_chunk` — write to disk only once every `file_chunk` steps
  (default: 10)
 
Returns: a vector of samples
"""   
function Sample(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    target::Target,
    n::Int;
    thinning::Int=1,
    init_params = nothing,
    file_chunk=10,
    fol_name = ".",
    file_name = "samples",
    progress = true,
    kwargs...,
)
    io = open(joinpath(fol_name, "VarNames.txt"), "w") do io
        println(io, string(target.θ_names))
    end

    ### initial conditions ###
    if init_params == nothing
        init_params = target.θ_start
    end
    trans_init_params = target.transform(init_params)

    transition, state = Step(
        rng,
        sampler,
        target.h,
        trans_init_params;
        bijector = target.inv_transform,
        kwargs...)

    sample = _make_sample(transition)
    samples = similar(sample, (length(sample), Int(floor(n/thinning))))
    samples[:, 1] .= sample

    pbar = Progress(n, (progress ? 0 : Inf), "MCHMC: ")

    write_chain(file_name, size(samples)..., eltype(sample), file_chunk) do chain_file
        for i in 2:n
            transition, state = Step(
                    rng,
                    sampler,
                    state;
                    bijector = target.transform,
                    kwargs...,
                )
            if mod(i, thinning)==0
                j = Int(floor(i/thinning))
                samples[:,j] = sample = _make_sample(transition)
                if chain_file !== nothing      
                    push!(chain_file, sample)
                end
            end
            ProgressMeter.next!(pbar, showvalues = [
                ("ϵ", sampler.hyperparameters.eps),
                ("dE/d", state.dE / target.d)
            ])
        end
    end

    ProgressMeter.finish!(pbar)

    return samples
end
