function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    spl::MCHMCSampler;
    init_params = nothing,
    kwargs...,
)
    logdensity = model.logdensity
    logdensity = LogDensityProblemsAD.ADgradient(logdensity)
    if init_params == nothing
        d = LogDensityProblems.dimension(logdensity)
        init_params = randn(rng, d)
    end
    h = Hamiltonian(logdensity)
    return Step(rng, spl, h, init_params; kwargs...)
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
