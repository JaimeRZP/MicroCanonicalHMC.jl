function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::MCHMCSampler;
    init_params = nothing,
    kwargs...,
)
    logdensity = model.logdensity
    logdensity = LogDensityProblemsAD.ADgradient(model.logdensity)
    if init_params == nothing
        d = LogDensityProblems.dimension(logdensity)
        init_params = randn(rng, d)
    end
    h = Hamiltonian(logdensity)
    return Step(rng, spl, h; trans_init_params=init_params, kwargs...)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::MCHMCSampler,
    state::MCHMCState;
    kwargs...,
)
    return Step(rng, sampler, state; kwargs...)
end
