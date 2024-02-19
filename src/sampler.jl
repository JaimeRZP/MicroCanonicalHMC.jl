mutable struct Hyperparameters{T}
    eps::T
    L::T
    nu::T
    lambda_c::T
    sigma::AbstractVector{T}
    gamma::T
    sigma_xi::T
end

Hyperparameters(; kwargs...) = begin
    eps = get(kwargs, :eps, 0.0)
    L = get(kwargs, :L, 0.0)
    nu = get(kwargs, :nu, 0.0)
    sigma = get(kwargs, :sigma, [0.0])
    lambda_c = get(kwargs, :lambda_c, 0.1931833275037836)
    gamma = get(kwargs, :gamma, (50 - 1) / (50 + 1)) #(neff-1)/(neff+1) 
    sigma_xi = get(kwargs, :sigma_xi, 1.5)
    Hyperparameters(eps, L, nu, lambda_c, sigma, gamma, sigma_xi)
end

mutable struct Settings{T}
    nadapt::Int
    TEV::T
    nchains::Int
    adaptive::Bool
    integrator::String
    init_eps::Union{Nothing,T}
    init_L::Union{Nothing,T}
    init_sigma::Union{Nothing,AbstractVector{T}}
end

Settings(; kwargs...) = begin
    kwargs = Dict(kwargs)
    nadapt = get(kwargs, :nadapt, 1000)
    TEV = get(kwargs, :TEV, 0.001)
    adaptive = get(kwargs, :adaptive, false)
    nchains = get(kwargs, :nchains, 1)
    integrator = get(kwargs, :integrator, "LF")
    init_eps = get(kwargs, :init_eps, nothing)
    init_L = get(kwargs, :init_L, nothing)
    init_sigma = get(kwargs, :init_sigma, nothing)
    Settings(nadapt, TEV, nchains, adaptive, integrator, init_eps, init_L, init_sigma)
end

struct MCHMCSampler <: AbstractMCMC.AbstractSampler
    settings::Settings
    hyperparameters::Hyperparameters
    hamiltonian_dynamics::Function
end


"""
    MCHMC(
        nadapt::Int,
        TEV::Real;
        kwargs...
    )

Constructor for the MicroCanonical HMC sampler
"""
function MCHMC(nadapt::Int, TEV::Real; kwargs...)
    """the MCHMC (q = 0 Hamiltonian) sampler"""
    sett = Settings(; nadapt = nadapt, TEV = TEV, kwargs...)
    hyperparameters = Hyperparameters(; kwargs...)

    ### integrator ###
    if sett.integrator == "LF" # leapfrog
        hamiltonian_dynamics = Leapfrog
    elseif sett.integrator == "MN" # minimal norm
        hamiltonian_dynamics = Minimal_norm
    else
        println(string("integrator = ", integrator, "is not a valid option."))
    end

    return MCHMCSampler(sett, hyperparameters, hamiltonian_dynamics)
end

function Random_unit_vector(rng::AbstractRNG, d::Int; T::Type=Float64, _normalize = true)
    """Generates a random (isotropic) unit vector."""
    u = randn(rng, T, d)
    if _normalize
        u = normalize(u)
    end
    return u
end

function Partially_refresh_momentum(rng::AbstractRNG, nu::Float64, u::AbstractVector)
    d = length(u)
    z = nu .* Random_unit_vector(rng, d; _normalize = false)
    uu = u .+ z
    return normalize(uu)
end

function Update_momentum(d::Number, eff_eps::Number, g::AbstractVector, u::AbstractVector)
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
    Feps::T
    Weps::T
    h::Hamiltonian
end

struct Transition{T}
    θ::Vector{T}
    ϵ::T
    δE::T
    ℓ::T
end

function Transition(state::MCHMCState{T}, bijector) where {T}
    eps = (state.Feps / state.Weps)^(-1 / 6)
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
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = length(init_params)
    l, g = -1 .* h.∂lπ∂θ(init_params)
    u = Random_unit_vector(rng, d; T=T)
    Weps = T(1e-5)
    Feps = Weps * sampler.hyperparameters.eps^(1/6)
    state = MCHMCState{T}(rng, 0, init_params, u, l, g, 0.0, Feps, Weps, h)
    state = tune_hyperparameters(rng, sampler, state; kwargs...)
    transition = Transition(state, bijector)
    return transition, state
end

function Step(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    state::MCHMCState;
    bijector = NoTransform,
    kwargs...,
)
    """One step of the Langevin-like dynamics."""
    dialog = get(kwargs, :dialog, false)

    eps = sampler.hyperparameters.eps
    nu = sampler.hyperparameters.nu
    sigma_xi = sampler.hyperparameters.sigma_xi
    gamma = get(kwargs, :gamma, sampler.hyperparameters.gamma)

    TEV = sampler.settings.TEV
    adaptive = get(kwargs, :adaptive, sampler.settings.adaptive)

    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, state)
    # Langevin-like noise
    uuu = Partially_refresh_momentum(rng, nu, uu)
    dEE = kinetic_change + ll - state.l

    if adaptive
        d = length(xx)
        varE = dEE^2 / d
        # 1e-8 is added to avoid divergences in log xi        
        xi = varE / TEV + 1e-8
        # the weight which reduces the impact of stepsizes which 
        # are much larger on much smaller than the desired one.        
        w = exp(-(1/2) * (log(xi) / (6 * sigma_xi))^2)
        # Kalman update the linear combinations
        Feps = gamma * state.Feps + w * (xi / eps^6)
        Weps = gamma * state.Weps + w
        new_eps = (Feps / Weps)^(-1 / 6)

        sampler.hyperparameters.eps = new_eps
        tune_nu!(sampler, d)
    else
        Feps = state.Feps
        Weps = state.Weps
    end

    state = MCHMCState(rng, state.i + 1, xx, uuu, ll, gg, dEE, Feps, Weps, state.h)
    transition = Transition(state, bijector)
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
function Sample(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    target::Target,
    n::Int;
    init_params = nothing,
    fol_name = ".",
    file_name = "samples",
    progress = true,
    kwargs...,
)
    io = open(joinpath(fol_name, "VarNames.txt"), "w") do io
        println(io, string(target.θ_names))
    end

    chain = []
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
    push!(chain, [transition.θ; transition.ϵ; transition.δE; transition.ℓ])

    io = open(joinpath(fol_name, string(file_name, ".txt")), "w") do io
        println(io, [transition.θ; transition.ϵ; transition.δE; transition.ℓ])
        @showprogress "MCHMC: " (progress ? 1 : Inf) for i = 1:n-1
            try
                transition, state = Step(
                    rng,
                    sampler,
                    state;
                    bijector = target.transform,
                    kwargs...,
                )
                push!(chain, [transition.θ; transition.ϵ; transition.δE; transition.ℓ])
                println(io, [transition.θ; transition.ϵ; transition.δE; transition.ℓ])
            catch err
                if err isa InterruptException
                    rethrow(err)
                else
                    @warn "Divergence encountered after tuning"
                end
            end
        end
    end

    io = open(joinpath(fol_name, string(file_name, "_summary.txt")), "w") do io
        ess, rhat = Summarize(chain)
        println(io, ess)
        println(io, rhat)
    end

    return chain
end
