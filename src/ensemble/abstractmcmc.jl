
function AbstractMCMC.step(
    sampler::EnsembleSampler,
    target::ParallelTarget,
    state;
    kwargs...,
)
    return Step(sampler::EnsembleSampler, target::ParallelTarget, state; kwargs...)
end

function AbstractMCMC.sample(
    model::DynamicPPL.Model,
    sampler::EnsembleSampler,
    N::Int;
    resume_from = nothing,
    kwargs...,
)

    if resume_from === nothing
        target = ParallelTarget(TuringTarget(model), sampler.settings.nchains)
        state, sample = Init(sampler, target; kwargs...)
        state, sample = tune_hyperparameters(sampler, target, state; kwargs...)
        init = (state, sample)
    else
        @info "Starting from previous run"
        target = resume_from.info[:target]
        sampler = resume_from.info[:sampler]
        init = resume_from.info[:init]
    end
    return AbstractMCMC.mcmcsample(target, sampler, init, N; kwargs...)
end

function AbstractMCMC.mcmcsample(
    target::ParallelTarget,
    sampler::EnsembleSampler,
    init,
    N::Integer;
    save_state = true,
    burn_in = 0,
    progress = PROGRESS[],
    progressname = "Chain 1",
    callback = nothing,
    thinning = 1,
    kwargs...,
)

    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    Ntotal = thinning * (N - 1) + burn_in + 1

    # Start the timer
    start = time()
    local state
    # Obtain the initial sample and state.
    state, sample = init

    AbstractMCMC.@ifwithprogresslogger progress name = progressname begin
        # Determine threshold values for progress logging
        # (one update per 0.5% of progress)
        if progress
            threshold = Ntotal ÷ 200
            next_update = threshold
        end

        # Run callback.
        callback === nothing || callback(rng, target, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, target, sampler, N; kwargs...)
        samples = AbstractMCMC.save!!(samples, sample, 1, target, sampler, N; kwargs...)

        # Update the progress bar.
        itotal = 1 + burn_in
        if progress && itotal >= next_update
            AbstractMCMC.ProgressLogging.@logprogress itotal / Ntotal
            next_update = itotal + threshold
        end

        # Step through the sampler.
        for i = 2:N
            # Discard thinned samples.
            for _ = 1:(thinning-1)
                # Obtain the next sample and state.
                state, sample = AbstractMCMC.step(sampler, target, state; kwargs...)

                # Update progress bar.
                if progress && (itotal += 1) >= next_update
                    AbstractMCMC.ProgressLogging.@logprogress itotal / Ntotal
                    next_update = itotal + threshold
                end
            end

            # Obtain the next sample and state.
            state, sample = AbstractMCMC.step(sampler, target, state; kwargs...)

            # Run callback.
            callback === nothing ||
                callback(rng, target, sampler, sample, state, i; kwargs...)

            # Save the sample.
            samples = AbstractMCMC.save!!(samples, sample, 1, target, sampler, N; kwargs...)

            # Update the progress bar.
            if progress && (itotal += 1) >= next_update
                AbstractMCMC.ProgressLogging.@logprogress itotal / Ntotal
                next_update = itotal + threshold
            end
        end
    end

    # Get the sample stop time.
    stop = time()
    duration = stop - start
    stats = AbstractMCMC.SamplingStats(start, stop, duration)

    return AbstractMCMC.bundle_samples(
        samples,
        target,
        sampler,
        state;
        save_state = save_state,
        stats = stats,
        burn_in = burn_in,
        thinning = thinning,
        kwargs...,
    )

end

function AbstractMCMC.bundle_samples(
    samples::Vector,
    target::ParallelTarget, #::AbstractMCMC.AbstractModel,
    sampler::EnsembleSampler, #::AbstractMCMC.AbstractSampler,
    state;
    save_state = true,
    stats = missing,
    burn_in = 0,
    thinning = 1,
    kwargs...,
)

    param_names = target.target.vsyms
    internal_names = [:E, :logp]
    names = [param_names; internal_names]

    # Set up the info tuple.
    if save_state
        info = (target = target, sampler = sampler, init = (state, samples[end]))
    else
        info = NamedTuple()
    end

    # Merge in the timing info, if available
    if !ismissing(stats)
        info = merge(info, (start_time = stats.start, stop_time = stats.stop))
    end

    # Conretize the array before giving it to MCMCChains.
    samples_matrix = zeros(length(samples), sampler.settings.nchains, target.target.d + 2)
    for i = 1:length(samples)
        X, E, L = samples[1]
        samples_matrix[i, :, :] = [X E L]
    end
    sequantial_samples = unroll_chains(samples_matrix)
    sequantial_samples = MCMCChains.concretize(sequantial_samples)

    # Chain construction.
    chain = MCMCChains.Chains(
        sequantial_samples,
        names,
        (internals = internal_names,);
        info = info,
        start = burn_in + 1,
        thin = thinning,
    )

    return chain
end
