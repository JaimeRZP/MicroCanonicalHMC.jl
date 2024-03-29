mutable struct Hyperparameters{T}
    eps::T
    L::T
    sigma::AbstractVector{T}
    lambda_c::T
    gamma::T
    sigma_xi::T
end

function Hyperparameters(; kwargs...)
    eps = get(kwargs, :eps, 0.0)
    L = get(kwargs, :L, 0.0)
    sigma = get(kwargs, :sigma, [0.0])
    lambda_c = get(kwargs, :lambda_c, 0.0)
    gamma = get(kwargs, :gamma, 0.0)
    sigma_xi = get(kwargs, :sigma_xi, 0.0)
    return Hyperparameters(eps, L, sigma, lambda_c, gamma, sigma_xi)
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

function Random_unit_vector(rng::AbstractRNG, x::AbstractVector{T}; _normalize = true) where {T}
    """Generates a random (isotropic) unit vector."""
    u = similar(x)
    randn!(rng, u)
    if _normalize
        u ./= norm(u)
    end
    return u
end

function Partially_refresh_momentum(rng::AbstractRNG, nu::T, u::AbstractVector{T}) where {T}
    d = length(u)
    z = nu .* Random_unit_vector(rng, u; _normalize = false)
    uu = u .+ z
    return uu ./ norm(uu)
end

function Update_momentum(d::Int, eff_eps::T, g::AbstractVector{T}, u::AbstractVector{T}) where {T}
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
    return uu ./ norm(uu), delta_r
end

struct MCHMCState{T}
    rng::AbstractRNG
    i::Int
    x::AbstractVector{T}
    u::AbstractVector{T}
    l::T
    g::AbstractVector{T}
    dE::T
    Weps::T
    Feps::T
    h::Hamiltonian
end

struct Transition{T}
    θ::AbstractVector{T}
    ϵ::T
    δE::T
    ℓ::T
end

Transition(state::MCHMCState) = Transition(state, NoTransform)

function Transition(state::MCHMCState{T}, inv_transform) where {T}
    eps = (state.Feps / state.Weps)^(-1 / 6)
    θ = inv_transform(state.x)
    return Transition(θ, T(eps), state.dE, -state.l)
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
    x::AbstractVector{T};
    inv_transform = NoTransform,
    kwargs...,
) where {T}
    kwargs = Dict(kwargs)
    d = length(x)
    l, g = -1 .* h.∂lπ∂x(x)
    u = Random_unit_vector(rng, x)
    eps = sampler.hyperparameters.eps
    Weps = T(1e-5)
    Feps = T(Weps * eps^(1 / 6))
    state = MCHMCState{T}(rng, 0, x, u, l, g, T(0.0), Weps, Feps, h)
    state = tune_hyperparameters(rng, sampler, state; kwargs...)
    transition = Transition(state, inv_transform)
    return transition, state
end

function Step(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    state::MCHMCState{T};
    inv_transform = NoTransform,
    kwargs...,
) where {T}
    """One step of the Langevin-like dynamics."""
    dialog = get(kwargs, :dialog, false)

    eps = sampler.hyperparameters.eps
    Feps = state.Feps
    Weps = state.Weps
    L = sampler.hyperparameters.L
    sigma_xi = sampler.hyperparameters.sigma_xi
    gamma = sampler.hyperparameters.gamma

    TEV = sampler.TEV

    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, state)
    # Langevin-like noise
    nu = eval_nu(eps, L, length(xx))
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
        Feps = T(gamma * Feps + w * (xi / eps^6))
        Weps = T(gamma * Weps + w)
        eps  = T((Feps / Weps)^(-1 / 6))
        sampler.hyperparameters.eps = eps
    end

    state = MCHMCState(rng, state.i + 1, xx, uuu, ll, gg, dEE, Weps, Feps, state.h)
    transition = Transition(state, inv_transform)
    return transition, state
end

function _make_sample(transition::Transition; transform=NoTransform, include_latent=false)
    if include_latent
        sample = [transition.θ[:]; transform(transition.θ)[:]; transition.ϵ; transition.δE; transition.ℓ]
    else
        sample = [transition.θ[:]; transition.ϵ; transition.δE; transition.ℓ]
    end
    return sample
end

function Sample(
    sampler::MCHMCSampler,
    target::Target,
    n::Int;
    kwargs...,
)
    return Sample(
        Random.GLOBAL_RNG,
        sampler,
        target,
        n;
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
    include_latent = false,
    kwargs...,
)
    io = open(joinpath(fol_name, "VarNames.txt"), "w") do io
        println(io, string(target.θ_names))
    end

    ### initial conditions ###
    if init_params == nothing
        θ_start = target.θ_start
    else 
        @info "Using provided init_params"
        θ_start = init_params
    end
    x_start = target.transform(θ_start)

    transition, state = Step(
        rng,
        sampler,
        target.h,
        x_start;
        inv_transform = target.inv_transform,
        kwargs...)

    sample = _make_sample(transition; transform=target.transform, include_latent=include_latent)
    samples = similar(sample, (length(sample), Int(floor(n/thinning))))
    samples[:,1] = sample 
    pbar = Progress(n; desc="Sampling: ")
    write_chain(file_name, size(samples)..., eltype(sample), file_chunk) do chain_file
        for i in 2:n
            transition, state = Step(
                    rng,
                    sampler,
                    state;
                    inv_transform = target.inv_transform,
                    kwargs...,
                )
            if mod(i, thinning)==0
                j = Int(floor(i/thinning))
                samples[:,j] = sample = _make_sample(transition;
                    transform=target.transform, include_latent=include_latent)
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
