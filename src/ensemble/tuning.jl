function tune_what(sampler::EnsembleSampler, target::ParallelTarget)
    tune_sigma, tune_eps, tune_L = false, false, false

    if sampler.hyperparameters.sigma == [0.0]
        @info "Tuning sigma ⏳"
        tune_sigma = true
        if sampler.settings.init_sigma == nothing
            init_sigma = ones(target.target.d)
        else
            init_sigma = sampler.settings.init_sigma
        end
        sampler.hyperparameters.sigma = init_sigma
    else
        @info "Using given sigma ✅"
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
    else
        @info "Using given eps ✅"
    end

    if sampler.hyperparameters.L == 0.0
        @info "Tuning L ⏳"
        tune_L = true
        if sampler.settings.init_sigma == nothing
            init_L = sqrt(target.target.d)
        else
            init_L = sampler.settings.init_L
        end
        sampler.hyperparameters.L = init_L
    else
        @info "Using given L ✅"
    end

    tune_nu!(sampler, target)
    @info string("Initial nu ", sampler.hyperparameters.nu)

    return tune_sigma, tune_eps, tune_L
end

function Summarize(samples::AbstractArray)
    _samples = convert(Array{Float64,3}, samples)
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)
    return ess, rhat
end

function tune_L!(
    sampler::EnsembleSampler,
    target::ParallelTarget,
    init::EnsembleState;
    kwargs...,
)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps

    nchains = sampler.settings.nchains
    d = target.target.d

    steps = 10 .^ (LinRange(2, log10(sett.tune_L_nsteps), 20))
    steps /= nchains
    steps = Int.(round.(steps))
    chains = zeros(sum(steps), nchains, d + 2)
    l = 0
    for s in steps
        l += s
        for i = 1:s
            init = Step(sampler, target, init; kwargs...)
            sample = [init.x init.dE -init.l]
            chains[i, :, :] = sample
        end
        neffs = Neff(chains, l * nchains)
        neff = mean(neffs)
        if dialog
            println(string("samples: ", l * nchains, "--> 1/<1/ess>: ", neff))
        end
        if (l * nchains) > (10.0 / neff)
            sampler.hyperparameters.L = 0.4 * eps / neff # = 0.4 * correlation length
            @info string("Found L: ", sampler.hyperparameters.L, " ✅")
            break
        end
    end
end

function Virial_loss(x::AbstractMatrix, g::AbstractMatrix)
    """loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    #should be all close to 1 if we have reached the typical set
    v = mean(x .* g, dims = 1)  # mean over params
    return sqrt.(mean((v .- 1.0) .^ 2))
end

function Step_burnin(
    sampler::EnsembleSampler,
    target::ParallelTarget,
    init::EnsembleState;
    kwargs...,
)
    dialog = get(kwargs, :dialog, false)
    step = Step(sampler, target, init)
    lloss = Virial_loss(step.x, step.g)
    return lloss, step
end

function Init_burnin(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    x = init.x
    l = init.l
    g = init.g
    dE = init.dE

    v = mean(x .* g, dims = 1)
    loss = mean((1 .- v) .^ 2)
    sng = -2.0 .* (v .< 1.0) .+ 1.0
    u = -g ./ sqrt.(sum(g .^ 2, dims = 2))
    u .*= sng

    if dialog
        println("Initial Virial loss: ", loss)
    end

    return loss, EnsembleState(x, u, l, g, dE)
end

function dual_averaging(
    sampler::EnsembleSampler,
    target::ParallelTarget,
    state;
    α = 1,
    kwargs...,
)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    varE_wanted = sett.varE_wanted
    d = target.target.d

    chains = zeros(sett.num_energy_points, sett.nchains, d + 2)
    for i = 1:sett.num_energy_points
        state, sample = Step(sampler, target, state; kwargs...)
        X, E, L = sample
        chains[i, :, :] = [X E L]
    end
    Es = vec(mean(chains[:, :, 3], dims = 2))
    m_E = mean(Es)
    s_E = std(Es)
    Es = deleteat!(Es, abs.(Es .- m_E) / s_E .> 2)
    varE = std(Es)^2 / d

    if dialog
        println("eps: ", sampler.hyperparameters.eps, " --> VarE: ", varE)
    end

    no_divergences = isfinite(varE)
    ### update the hyperparameters ###
    if no_divergences
        success = (abs(varE - varE_wanted) / varE_wanted) < 0.1
        if !success
            new_log_eps = log(sampler.hyperparameters.eps) - α * (varE - varE_wanted)
            sampler.hyperparameters.eps = exp(new_log_eps)
        end
    else
        success = false
        sampler.hyperparameters.eps *= 0.5
    end

    return success
end

function adaptive_step(
    sampler::EnsembleSampler,
    target::ParallelTarget,
    init;
    sigma_xi::Float64 = 1.0,
    gamma::Float64 = (50 - 1) / (50 + 1), # (neff-1)/(neff+1) 
    kwargs...,
)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    varE_wanted = sett.varE_wanted
    d = target.target.d

    step, Feps, Weps, max_eps = init
    step = Step(sampler, target, step)

    varE = mean(step.dE .^ 2) / d
    if dialog
        println("eps: ", eps, " --> VarE/d: ", varE)
    end

    no_divergences = isfinite(varE)
    if no_divergences
        success = (abs(varE - varE_wanted) / varE_wanted) < 0.05
        if !success
            xi = varE / varE_wanted + 1e-8
            w = exp(-0.5 * (log(xi) / (6.0 * sigma_xi))^2)
            # Kalman update the linear combinations
            Feps = gamma * Feps + w * (xi / eps^6)
            Weps = gamma * Weps + w
            new_eps = (Feps / Weps)^(-1 / 6)

            if new_eps > max_eps
                sampler.hyperparameters.eps = max_eps
            else
                sampler.hyperparameters.eps = new_eps
            end
        else
            @info string("Found eps: ", sampler.hyperparameters.eps, " ✅")
        end
    else
        success = false
        max_eps = sampler.hyperparameters.eps
        sampler.hyperparameters.eps = 0.5 * eps
    end

    return success, (step, Feps, Weps, max_eps)
end

function tune_nu!(sampler::EnsembleSampler, target::ParallelTarget)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    d = target.target.d
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function tune_hyperparameters(
    sampler::EnsembleSampler,
    target::ParallelTarget,
    init::EnsembleState;
    burn_in::Int = 0,
    kwargs...,
)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    d = target.target.d

    tune_sigma, tune_eps, tune_L = tune_what(sampler, target)

    if burn_in > 0
        @info "Starting burn in ⏳"
        loss, init = Init_burnin(sampler, target, init; kwargs...)
        xs = init.x
        for i = 1:burn_in
            lloss, step = Step_burnin(sampler, target, init; kwargs...)
            if lloss < loss
                if dialog
                    println(
                        "Virial loss: ",
                        lloss,
                        " --> Relative improvement: ",
                        abs(lloss / loss - 1),
                    )
                end
                xs = [xs; init.x]
                if (lloss <= sampler.settings.loss_wanted) || (abs(lloss / loss - 1) < 0.01)
                    @info string("Virial loss condition met during burn-in at step: ", i)
                    break
                end
                loss = lloss
                init = step
            else
                uu = Partially_refresh_momentum(sampler, target, init.u)
                init = EnsembleState(init.x, uu, init.l, init.g, init.dE)
            end
            if i == burn_in
                @warn "Maximum number of steps reached during burn-in"
            end
        end
        if tune_sigma
            sigma = vec(std(xs, dims = 1))
            sigma ./= sqrt(sum(sigma .^ 2))
            sampler.hyperparameters.sigma = sigma
            @info string("Found sigma: ", sampler.hyperparameters.sigma, " ✅")
        end
    end

    if tune_eps
        tuning_method = get(kwargs, :tuning_method, "AdaptiveStep")
        if dialog
            println(string("Using eps tuning method ", tuning_method))
        end
        if tuning_method == "DualAveraging"
            for i = 1:sett.tune_eps_nsteps
                success = dual_averaging(
                    sampler,
                    target,
                    init;
                    α = exp.(-(i .- 1) / 20),
                    kwargs...,
                )
                if success
                    break
                end
            end
        end
        if tuning_method == "AdaptiveStep"
            Weps = 1e-5
            Feps = Weps * sampler.hyperparameters.eps^(1 / 6)
            tuning_init = (init, Feps, Weps, Inf)
            for i = 1:sett.tune_eps_nsteps
                success, tuning_init =
                    adaptive_step(sampler, target, tuning_init; kwargs...)
                if success
                    break
                end
            end
        end
    end

    if tune_L
        tune_L!(sampler, target, init; kwargs...)
    end

    tune_nu!(sampler, target)
    @info string("Final nu ", sampler.hyperparameters.nu)

    return init
end
