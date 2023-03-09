mutable struct EnsembleSettings
    nchains::Int
    key::MersenneTwister
    loss_wanted::Float64
    varE_wanted::Float64
    VarE_maxiter::Int
    num_energy_points::Int
    integrator::String
    init_eps
    init_L
    init_sigma
end

EnsembleSettings(;kwargs...) = begin
    kwargs = Dict(kwargs)
    seed = get(kwargs, :seed, 0)
    key = MersenneTwister(seed)
    nchains = get(kwargs, :nchains, 1)
    loss_wanted = get(kwargs, :loss_wanted, 1.0)
    varE_wanted = get(kwargs, :varE_wanted, 0.01)
    VarE_maxiter = get(kwargs, :varE_maxiter, 100)
    num_energy_points = get(kwargs, :num_energy_points, 20)
    integrator = get(kwargs, :integrator, "LF")
    init_eps = get(kwargs, :init_eps, nothing)
    init_L = get(kwargs, :init_L, nothing)
    init_sigma = get(kwargs, :init_sigma, nothing)
    EnsembleSettings(nchains, key,
             loss_wanted, varE_wanted, VarE_maxiter, num_energy_points,
             integrator, init_eps, init_L, init_sigma)
end

struct EnsembleSampler <: AbstractMCMC.AbstractSampler
   settings::EnsembleSettings
   hyperparameters::Hyperparameters
   hamiltonian_dynamics::Function
end

function MCHMC(eps, L, nchains; kwargs...)

   sett = EnsembleSettings(;nchains=nchains, kwargs...)
   hyperparameters = Hyperparameters(;eps=eps, L=L, kwargs...)

   if sett.integrator == "LF"  # leapfrog
       hamiltonian_dynamics = Leapfrog
       grad_evals_per_step = 1.0
   elseif sett.integrator == "MN"  # minimal norm integrator
       hamiltonian_dynamics = Minimal_norm
       grad_evals_per_step = 2.0
   else
       println(string("integrator = ", integrator, "is not a valid option."))
   end

   return EnsembleSampler(sett, hyperparameters, hamiltonian_dynamics)
end

function MCHMC(nchains; kwargs...)
    return MCHMC(0.0, 0.0, nchains; kwargs...)
end

function Random_unit_vector(sampler::EnsembleSampler, target::ParallelTarget; normalize=true)
    """Generates a random (isotropic) unit vector."""
    sett = sampler.settings
    return Random_unit_matrix(sett.key, sett.nchains, target.target.d; normalize=normalize)
end

function Random_unit_matrix(key, nchains, d; normalize = true)
    u = randn(key, nchains, d)
    if normalize
        u ./= sqrt.(sum(u.^2, dims=2))
    end
    return u
end

function Partially_refresh_momentum(sampler::EnsembleSampler, target::ParallelTarget, u)
    """Adds a small noise to u and normalizes."""
    return Partially_refresh_momentum(sampler.hyperparameters.nu, sampler.settings.key,
                                      sampler.settings.nchains, target.target.d, u; normalize = false)
end

function Partially_refresh_momentum(nu, key, nchains, d, u; normalize = false)
    z = nu .* Random_unit_matrix(key, nchains, d; normalize=normalize)
    uu = (u .+ z) ./ sqrt.(sum((u .+ z).^2, dims=2))
    return uu
end

function Update_momentum(target::ParallelTarget, eff_eps::Number,
                         g::AbstractMatrix, u::AbstractMatrix)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
    Update_momentum(target.target.d, eff_eps::Number, g ,u)
end

function Update_momentum(d::Number, eff_eps::Number,
                         g::AbstractMatrix ,u::AbstractMatrix)
    g_norm = sqrt.(sum(g .^2, dims=2))[:, 1]
    e = - g ./ g_norm
    # Matrix dot product along dim=2
    ue = sum(u .* e, dims=2)[:, 1]
    sh = sinh.(eff_eps .* g_norm ./ d)
    ch = cosh.(eff_eps .* g_norm ./ d)
    th = tanh.(eff_eps .* g_norm ./ d)
    delta_r = log.(ch) .+ log1p.(ue .* th)

    uu = @.((u + e * (sh + ue * (ch - 1))) / (ch + ue * sh))

    return uu, delta_r
end

function Dynamics(sampler::EnsembleSampler, target::ParallelTarget, state)
    """One step of the Langevin-like dynamics."""
    x, u, l, g, dE = state
    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, target, x, u, l, g)
    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, target, uu)
    dEE = kinetic_change .+ ll .- l
    return xx, uuu, ll, gg, dEE
end

function Init(sampler::EnsembleSampler, target::ParallelTarget; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = target.target.d
    ### initial conditions ###
    if :initial_x ∈ keys(kwargs)
        x = target.transform(kwargs[:initial_x])
    else
        x = target.prior_draw(sett.key)
    end
    l, g = target.nlogp_grad_nlogp(x)
    g .*= d/(d-1)
    u = Random_unit_vector(sampler, target) #random initial direction

    dE = zeros(sett.nchains)
    sample = (target.inv_transform(x), dE, -l)
    state = (x, u, l, g, dE)
    return state, sample
end

function Step(sampler::EnsembleSampler, target::ParallelTarget, state; kwargs...)
    """Tracks transform(x) as a function of number of iterations"""
    sett = sampler.settings
    x, u, l, g, dE = state
    step = Dynamics(sampler, target, state)
    xx, uu, ll, gg, dEE = step

    return step, (target.inv_transform(xx), dE .+ dEE, -ll)
end


function Sample(sampler::EnsembleSampler, target::Target,
                num_steps::Int, burnin::Int;
                remove_initial=0,  kwargs...)
    """Args:
           num_steps: number of integration steps to take.
           x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
           random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
        Returns:
            samples (shape = (num_steps, self.Target.d))
    """
    nchains = sampler.settings.nchains
    target = ParallelTarget(target, nchains)

    state, sample = Init(sampler, target; kwargs...)
    state, sample = Burnin(sampler, target, state, burnin; kwargs...)

    d = target.target.d
    chains = zeros(num_steps, nchains, d+2)
    X, E, L = sample
    chains[1, :, :] = [X E L]
    for i in 2:num_steps
        state, sample = Step(sampler, target, state; kwargs...)
        X, E, L = sample
        chains[i, :, :] = [X E L]
    end

    return unroll_chains(chains; burnin=remove_initial+1)
end

function unroll_chains(chains; burnin=1)
    chains = chains[burnin:end, :, :]
    nsteps, nchains, nparams = axes(chains)
    chain = []
    for i in nchains
        A = chains[:, i, :]
        chain_i = Vector{eltype(A)}[eachrow(A)...]
        chain = cat(chain, chain_i; dims=1)
    end
    return chain
end