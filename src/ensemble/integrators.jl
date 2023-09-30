function Leapfrog(sampler::EnsembleSampler, target::ParallelTarget, state::EnsembleState)
    eps = sampler.hyperparameters.eps
    sigma = sampler.hyperparameters.sigma
    return Leapfrog(target, eps, sigma, state.x, state.u, state.l, state.g)
end

function Leapfrog(
    target::ParallelTarget,
    eps::Number,
    sigma::AbstractVector,
    x::AbstractMatrix,
    u::AbstractMatrix,
    l::AbstractVector,
    g::AbstractMatrix,
)
    sigma = hcat(sigma)'
    d = target.target.d
    uu, dr1 = Update_momentum(target, eps * 0.5, g .* sigma, u)

    #full step in x
    z = x ./ sigma # go to the latent space
    zz = z .+ eps .* uu

    xx = zz .* sigma # rotate back to parameter space
    ll, gg = target.nlogp_grad_nlogp(xx)
    gg .*= d / (d - 1)

    #half step in momentum
    uu, dr2 = Update_momentum(target, eps * 0.5, gg .* sigma, uu)

    kinetic_change = (dr1 .+ dr2) .* d

    return xx, uu, ll, gg, kinetic_change
end

function Minimal_norm(
    sampler::EnsembleSampler,
    target::ParallelTarget,
    state::EnsembleState,
)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    # V T V T V
    eps = sampler.hyperparameters.eps
    lambda_c = sampler.hyperparameters.lambda_c
    sigma = sampler.hyperparameters.sigma

    return Minimal_norm(target, eps, lambda_c, state.x, state.u, state.l, state.g)
end

function Minimal_norm(
    target::ParallelTarget,
    eps::Number,
    lambda_c::Number,
    sigma::AbstractVector,
    x::AbstractMatrix,
    u::AbstractMatrix,
    l::AbstractVector,
    g::AbstractMatrix,
)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    # V T V T V
    sigma = hcat(sigma)'
    d = target.target.d
    uu, dr1 = Update_momentum(d, eps * lambda_c, g .* sigma, u)

    z = x ./ sigma # go to the latent space
    zz = z .+ eps .* 0.5 .* uu
    xx = zz .* sigma
    ll, gg = target.nlogp_grad_nlogp(xx)
    gg .*= d / (d - 1)

    uu, dr2 = Update_momentum(d, eps * (1 - 2 * lambda_c), gg .* sigma, uu)

    zz = zz .+ eps .* 0.5 .* uu
    xx = zz .* sigma
    ll, gg = target.nlogp_grad_nlogp(xx)
    gg .*= d / (d - 1)

    uu, dr3 = Update_momentum(d, eps * lambda_c, gg .* sigma, uu)

    kinetic_change = (dr1 .+ dr2 .+ dr3) .* (d - 1)

    return xx, uu, ll, gg, kinetic_change
end
