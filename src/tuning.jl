function tune_what!(sampler::MCHMCSampler)
    if sampler.tune_eps
        @info "Tuning eps ⏳"
    end
    if sampler.tune_L
        @info "Tuning L ⏳"
    end
    if sampler.tune_sigma
        @info "Tuning sigma ⏳"
    end
end

function init_hyperparameters!(sampler::MCHMCSampler, T::Type, d::Int)
    init_hp = sampler.hyperparameters

    if init_hp.eps == 0.0
        eps = T((1/2))
    else
        eps = T(init_hp.eps)
    end

    if init_hp.L == 0.0
        L = T(sqrt(d))
    else
        L = T(init_hp.L)
    end

    if init_hp.sigma == [0.0]
        sigma = ones(T, d)
    else
        sigma = T.(init_hp.sigma)
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

    new_hp = Hyperparameters(eps, L, sigma, lambda_c, gamma, sigma_xi)
    sampler.hyperparameters = new_hp
end


function eval_nu(eps, L, d)
    nu = sqrt((exp(2 * eps / L) - 1) / d)
    return nu
end

function tune_hyperparameters(
    rng::AbstractRNG,
    sampler::MCHMCSampler,
    state::MCHMCState{T};
    kwargs...,
) where {T}
    ### Init Hyperparameters ###
    d = length(state.x)
    tune_what!(sampler)
    init_hyperparameters!(sampler, T, d)

    # Tuning
    xs = adapt(Array, state.x[:]) 
    window = Int(sampler.nadapt / 10)
    pbar = Progress(sampler.nadapt; desc="Tuning: ")
    for i = 1:sampler.nadapt
        _, state = Step(rng, sampler, state; adaptive = sampler.tune_eps, kwargs...)
        xs = [xs adapt(Array, state.x[:])]
        if mod(i, Int(sampler.nadapt / window)) == 0
            sigma = vec(std(xs, dims = 2))
            if sampler.tune_sigma
                sampler.hyperparameters.sigma = sigma
            end
            if sampler.tune_L
                ess, _ = Summarize(xs)
                m_ess = mean(ess)
                if m_ess > length(xs)/50
                    l = length(xs)/m_ess
                    eps = sampler.hyperparameters.eps
                    sampler.hyperparameters.L = 0.4*eps*l
                else
                    @warn "Effective sample size is too low, using sigma to tune L"
                    sampler.hyperparameters.L =
                        sqrt(mean(sigma .^ 2)) * sampler.hyperparameters.eps
                end
            end
        end
        ProgressMeter.next!(pbar, showvalues = [
            ("ϵ", sampler.hyperparameters.eps),
            ("L", sampler.hyperparameters.L),
            ("dE/d", state.dE / d)
        ])
    end
    ProgressMeter.finish!(pbar)
    return state
end
