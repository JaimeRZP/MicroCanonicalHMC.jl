function init_hyperparameters!(sampler::MCHMCSampler, T::Type, d::Int)
    init_hp = sampler.hyperparameters
    println("Type: ", T)
    if sampler.tune_sigma
        @info "Tuning sigma ⏳"
        if init_hp.sigma == [0.0]
            sigma = ones(T, d)
        else
            sigma = T.(init_hp.sigma)
        end
    end

    if sampler.tune_eps
        @info "Tuning eps ⏳"
        if init_hp.eps == 0.0
            eps = T((1/2) * sqrt(d))
        else
            eps = T(init_hp.eps)
        end
    end

    if sampler.tune_L
        @info "Tuning L ⏳"
        if init_hp.L == 0.0
            L = T(sqrt(d))
        else
            L = T(init_hp.L)
        end
    end

    if init_hp.lambda_c == 0.0
        lambda_c = T(0.1931833275037836)
    else
        lambda_c = T(init_hp.lambda_c)
    end

    if init_hp.gamma == 0.0
        gamma = T((50 - 1) / (50 + 1)) #(neff-1)/(neff+1) 
    else
        gamma = T(init_hp.gamma)
    end

    if init_hp.sigma_xi == 0.0
        sigma_xi = T(1.5)
    else
        sigma_xi = T(init_hp.sigma_xi)
    end

    nu = eval_nu(eps, L, d)
    new_hp = Hyperparameters(eps, L, nu, lambda_c, sigma, gamma, sigma_xi)
    sampler.hyperparameters = new_hp
end


function eval_nu(eps, L, d)
    nu = sqrt((exp(2 * eps / L) - 1) / d)
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
    state::MCHMCState{T};
    progress = true,
    kwargs...,
) where {T}
    ### Init Hyperparameters ###
    d = length(state.x)
    init_hyperparameters!(sampler, T, d)

    # Tuning
    xs = state.x[:]
    @showprogress "MCHMC (tuning): " (progress ? 1 : Inf) for i = 2:sampler.nadapt
        _, state = Step(rng, sampler, state; adaptive = sampler.tune_eps, kwargs...)
        xs = [xs state.x[:]]
        if mod(i, Int(sampler.nadapt / 5)) == 0
            sigma = vec(std(xs, dims = 2))
            if sampler.tune_sigma
                sampler.hyperparameters.sigma = sigma
            end
            if sdampler.tune_L
                sampler.hyperparameters.L =
                    sqrt(mean(sigma .^ 2)) * sampler.hyperparameters.eps
            end
        end
    end

    @info string("eps: ", sampler.hyperparameters.eps)
    @info string("L: ", sampler.hyperparameters.L)
    @info string("nu: ", sampler.hyperparameters.nu)
    @info string("sigma: ", sampler.hyperparameters.sigma)
    @info string("adaptive: ", sampler.adaptive)

    return state
end
