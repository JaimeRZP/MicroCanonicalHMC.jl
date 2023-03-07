function tune_what(sampler::Sampler, target::Target)
    tune_sigma, tune_eps, tune_L = false, false, false

    if sampler.hyperparameters.sigma == [0.0]
        @info "Tuning sigma ⏳"
        tune_sigma = true
        if sampler.settings.init_sigma == nothing
            init_sigma = ones(target.d)
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
            init_L = sqrt(target.d)
        else
            init_L = sampler.settings.init_L
        end
        sampler.hyperparameters.L = init_L
    end

    return tune_sigma, tune_eps, tune_L
end

function ess_corr(target::Target, samples)
    param_names = target.vsyms
    internal_names = [:E, :logp]
    names = [param_names; internal_names]
    samples = MCMCChains.concretize(samples)
    chain = MCMCChains.Chains(samples, names, (internals = internal_names,))
    stats = summarize(chain)
    esss = stats[:, :ess]  # effective sample size in each dimension

    ### my part (combine all dimensions): ###
    neff = esss ./ length(samples)
    return 1.0 / mean(1 ./ neff)
end

function tune_L!(sampler::Sampler, target::Target, init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    sampler.hyperparameters.L = sqrt(target.d)
    steps = 10 .^ (LinRange(2, log10(2500), sett.tune_maxiter))
    steps = Int.(round.(steps))
    samples = []
    for s in steps
        for i in 1:s
            init, sample = Step(sampler, target, init; monitor_energy=true)
            push!(samples, sample)
        end
        ESS = ess_corr(target, samples)
        if dialog
            println(string("samples: ", length(samples), "--> ESS: ", ESS))
        end
        if length(samples) > 10.0 / ESS
            sampler.hyperparameters.L = 0.4 * eps / ESS # = 0.4 * correlation length
            @info string("Found L: ", sampler.hyperparameters.L, " ✅")
            break
        end
    end
end

function tune_sigma!(sampler::Sampler, target::Target; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    MAP_t = target.MAP_t
    Hess = target.hess_nlogp(MAP_t)
    mass_mat = pinv(Hess)
    sigma = sqrt.(diag(mass_mat))
    sampler.hyperparameters.sigma = sigma
    @info string("Found sigma: ", sampler.hyperparameters.sigma, " ✅")
end

function tune_eps!(sampler::Sampler, target::Target, init; α=1, kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    L = sqrt(target.d) #sampler.hyperparameters.L
    varE_wanted = sett.varE_wanted
    #x, u, g, time = init

    samples = []
    for i in 1:sett.tune_samples
        init, sample = Step(sampler, target, init;
                            monitor_energy=true)
        push!(samples, sample)
    end

    samples = mapreduce(permutedims, vcat, samples)
    E = samples[:, end-1]
    varE = std(E)^2 / target.d #variance per dimension
    if dialog
        println("eps: ", eps, " --> VarE: ", varE)
    end
    no_divergences = isfinite(varE)

    ### update the hyperparameters ###
    if no_divergences
        success = (abs(varE-varE_wanted)/varE_wanted) < 0.05
        if !success

            new_log_eps = log(sampler.hyperparameters.eps)-α*(varE-varE_wanted)
            sampler.hyperparameters.eps = exp(new_log_eps)
        else
            @info string("Found eps: ", sampler.hyperparameters.eps, " ✅")
        end
    else
        success = false
        sampler.hyperparameters.eps = 0.5 * eps
    end

    return success
end

function eval_nu(eps, L, d)
    nu = sqrt((exp(2 * eps / L) - 1.0) / d)
    return nu
end

function tune_nu!(sampler::Sampler, target::Target)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    d = target.d
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function tune_hyperparameters(sampler::Sampler, target::Target, init; kwargs...)
    sett = sampler.settings
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)

    tune_sigma, tune_eps, tune_L = tune_what(sampler, target)

    if tune_sigma
        tune_sigma!(sampler, target; kwargs...)
    end

    if tune_eps
        for i in 1:sett.tune_maxiter
            α = exp.(-(i .- 1)/20)
            if tune_eps!(sampler, target, init; α=α, kwargs...)
                break
            end
        end
    end

    if tune_L
        tune_L!(sampler, target, init; kwargs...)
    end

    tune_nu!(sampler, target)
end
