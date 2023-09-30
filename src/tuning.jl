function tune_what(sampler::MCHMCSampler, d::Int)
    tune_sigma, tune_eps, tune_L = false, false, false

    if sampler.hyperparameters.sigma == [0.0]
        @info "Tuning sigma ⏳"
        tune_sigma = true
        if sampler.settings.init_sigma == nothing
            init_sigma = ones(d)
        else
            init_sigma = sampler.settings.init_sigma
        end
        sampler.hyperparameters.sigma = init_sigma
    end

    if sampler.hyperparameters.eps == 0.0
        @info "Tuning eps ⏳"
        tune_eps = true
        if sampler.settings.init_eps == nothing
            init_eps = 0.5
        else
            init_eps = sampler.settings.init_eps
        end
        sampler.hyperparameters.eps = init_eps
    end

    if sampler.hyperparameters.L == 0.0
        @info "Tuning L ⏳"
        tune_L = true
        if sampler.settings.init_sigma == nothing
            init_L = sqrt(d)
        else
            init_L = sampler.settings.init_L
        end
        sampler.hyperparameters.L = init_L
    end

    tune_nu!(sampler, d)

    return tune_sigma, tune_eps, tune_L
end

function Summarize(samples::AbstractVector)
    _samples = zeros(length(samples), 1, length(samples[1]))
    _samples[:, 1, :] = mapreduce(permutedims, vcat, samples)
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)
    return ess, rhat
end

function Summarize(samples::AbstractMatrix)
    dim_a, dim_b = size(samples)
    _samples = zeros(dim_a, 1, dim_b)
    _samples[:, 1, :] = samples
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)
    return ess, rhat
end

function Neff(samples, l::Int)
    ess, rhat = Summarize(samples)
    neff = ess ./ l
    return 1.0 / mean(1 ./ neff)
end

function eval_nu(eps, L, d)
    nu = sqrt((exp(2 * eps / L) - 1.0) / d)
    return nu
end

function tune_nu!(sampler::MCHMCSampler, d::Int)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function tune_hyperparameters(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    state::MCHMCState;
    progress = true,
    kwargs...,
)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings

    # Tuning
    d = length(state.x)
    tune_sigma, tune_eps, tune_L = tune_what(sampler, d)
    nadapt = sampler.settings.nadapt

    xs = state.x[:]
    @showprogress "MCHMC (tuning): " (progress ? 1 : Inf) for i = 2:nadapt
        _, state = Step(rng, sampler, state; adaptive = tune_eps, kwargs...)
        xs = [xs state.x[:]]
        if mod(i, Int(nadapt / 5)) == 0
            if dialog
                println(string("Burn in step: ", i))
                println(string("eps --->", sampler.hyperparameters.eps))
            end
            sigma = vec(std(xs, dims = 2))
            if tune_sigma
                sampler.hyperparameters.sigma = sigma
            end
            if tune_L
                sampler.hyperparameters.L =
                    sqrt(mean(sigma .^ 2)) * sampler.hyperparameters.eps
                if dialog
                    println(string("L   --->", sampler.hyperparameters.L))
                    println(" ")
                end
            end
        end
    end

    @info string("eps: ", sampler.hyperparameters.eps)
    @info string("L: ", sampler.hyperparameters.L)
    @info string("nu: ", sampler.hyperparameters.nu)
    @info string("sigma: ", sampler.hyperparameters.sigma)
    @info string("adaptive: ", sampler.settings.adaptive)

    return state
end
