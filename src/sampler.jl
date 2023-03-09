mutable struct Hyperparameters
    eps::Float64
    L::Float64
    nu::Float64
    lambda_c::Float64
    sigma::AbstractVector
end

Hyperparameters(;kwargs...) = begin
   eps = get(kwargs, :eps, 0.0)
   L = get(kwargs, :L, 0.0)
   nu = get(kwargs, :nu, 0.0)
   lambda_c = get(kwargs, :lambda_c, 0.1931833275037836)
   sigma = get(kwargs, :sigma, [0.0])
   Hyperparameters(eps, L, nu, lambda_c, sigma)
end

mutable struct Settings
    key::MersenneTwister
    varE_wanted::Float64
    burn_in::Int
    tune_samples::Int
    tune_maxiter::Int
    integrator::String
    init_eps
    init_L
    init_sigma
end

Settings(;kwargs...) = begin
    kwargs = Dict(kwargs)
    seed = get(kwargs, :seed, 0)
    key = MersenneTwister(seed)
    varE_wanted = get(kwargs, :varE_wanted, 0.2)
    burn_in = get(kwargs, :burn_in, 0)
    tune_samples = get(kwargs, :tune_samples, 1000)
    tune_maxiter = get(kwargs, :tune_maxiter, 100)
    integrator = get(kwargs, :integrator, "LF")
    init_eps = get(kwargs, :init_eps, nothing)
    init_L = get(kwargs, :init_L, nothing)
    init_sigma = get(kwargs, :init_sigma, nothing)
    Settings(key,
             varE_wanted, burn_in, tune_samples, tune_maxiter,
             integrator, init_eps, init_L, init_sigma)
end

struct Sampler <: AbstractMCMC.AbstractSampler
   settings::Settings
   hyperparameters::Hyperparameters
   hamiltonian_dynamics::Function
end

function MCHMC(eps, L; kwargs...)

   sett = Settings(;kwargs...)
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

   return Sampler(sett, hyperparameters, hamiltonian_dynamics)
end

function Random_unit_vector(sampler::Sampler, target::Target; normalize=true)
    """Generates a random (isotropic) unit vector."""
    return Random_unit_vector(sampler.settings.key, target.d; normalize=normalize)
end

function MCHMC(; kwargs...)
    return MCHMC(0.0, 0.0; kwargs...)
end

function Random_unit_vector(key, d; normalize = true)
    u = randn(key, d)
    if normalize
        u ./= sqrt(sum(u.^2))
    end
    return u
end

function Partially_refresh_momentum(sampler::Sampler, target::Target, u::AbstractVector)
    """Adds a small noise to u and normalizes."""
    return Partially_refresh_momentum(sampler.hyperparameters.nu, sampler.settings.key,
                                      target.d, u; normalize = false)
end

function Partially_refresh_momentum(nu, key, d, u::AbstractVector; normalize = false)
    z = nu .* Random_unit_vector(key, d; normalize=normalize)
    uu = (u .+ z) ./ sqrt(sum((u .+ z).^2))
    return uu
end

function Update_momentum(target::Target, eff_eps::Number,
                         g::AbstractVector, u::AbstractVector)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
    Update_momentum(target.d, eff_eps::Number,g ,u)
end

function Update_momentum(d::Number, eff_eps::Number,
                         g::AbstractVector, u::AbstractVector)
    g_norm = sqrt(sum(g .^2 ))
    e = - g ./ g_norm
    ue = dot(u, e)
    sh = sinh(eff_eps * g_norm / d)
    ch = cosh(eff_eps * g_norm / d)
    th = tanh(eff_eps * g_norm / d)
    delta_r = log(ch) + log1p(ue * th)

    uu = (u .+ e .* (sh + ue * (ch - 1))) / (ch + ue * sh)

    return uu, delta_r
end

function Dynamics(sampler::Sampler, target::Target, state)
    """One step of the Langevin-like dynamics."""
    x, u, l, g, dE = state
    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, target, x, u, l, g)
    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, target, uu)
    dEE = kinetic_change + ll - l
    return xx, uuu, ll, gg, dEE
end

function Energy(target::Target,
                x::AbstractVector, xx::AbstractVector,
                E::Number, kinetic_change::Number)
    l = target.nlogp(x)
    ll = target.nlogp(xx)
    dE = kinetic_change + ll - l
    EE = E + dE
    return -nllogp, EE
end

function Init(sampler::Sampler, target::Target; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = target.d
    ### initial conditions ###
    if :initial_x ∈ keys(kwargs)
        x = target.transform(kwargs[:initial_x])
    else
        x = target.prior_draw(sett.key)
    end
    l, g = target.nlogp_grad_nlogp(x)
    g .*= d/(d-1)
    u = Random_unit_vector(sampler, target) #random initial direction

    sample = [target.inv_transform(x); 0.0; -l]
    state = (x, u, l, g, 0.0)
    return state, sample
end

function Step(sampler::Sampler, target::Target, state; kwargs...)
    """Tracks transform(x) as a function of number of iterations"""
    x, u, l, g, dE = state
    step = Dynamics(sampler, target, state)
    xx, uu, ll, gg, dEE = step
    return step, [target.inv_transform(xx); dE + dEE; -ll]
end


function Sample(sampler::Sampler, target::Target, num_steps::Int;
                file_name="samples", progress=true, kwargs...)
    """Args:
           num_steps: number of integration steps to take.
           x_initial: initial condition for x (an array of shape (target dimension, )).
                      It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
           random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
        Returns:
            samples (shape = (num_steps, self.Target.d))
    """

    state, sample = Init(sampler, target; kwargs...)
    tune_hyperparameters(sampler, target, state; kwargs...)

    for i in 1:sampler.settings.burn_in
        state, sample = Step(sampler, target, state)
    end

    samples = []
    push!(samples, sample)
    io = open(string(file_name, ".txt"), "w") do io
        println(io, sample)
        for i in 1:num_steps
            state, sample = Step(sampler, target, state; kwargs...)
            push!(samples, sample)
            println(io, sample)
        end
    end

    return samples
end
