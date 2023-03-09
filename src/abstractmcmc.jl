
const PROGRESS = Ref(true)

function AbstractMCMC.step(sampler::Sampler, target::Target, state; kwargs...)
    return Step(sampler::Sampler, target::Target, state; kwargs...)
end

function AbstractMCMC.sample(model::DynamicPPL.Model,
                             sampler::Sampler, #AbstractMCMC.AbstractSampler,
                             N::Int;
                             resume_from=nothing,
                             kwargs...)

    if resume_from === nothing
        target = TuringTarget(model)
        init = Init(sampler, target; kwargs...)
        state, sample = init
        tune_hyperparameters(sampler, target, state; kwargs...)
    else
        @info "Starting from previous run"
        target = resume_from.info[:target]
        sampler = resume_from.info[:sampler]
        init = resume_from.info[:init]
    end
    return AbstractMCMC.mcmcsample(target, sampler, init, N; kwargs...)

end

function AbstractMCMC.mcmcsample(target::AbstractMCMC.AbstractModel,
                                 sampler::Sampler, #AbstractMCMC.AbstractSampler,
                                 init,
                                 N::Integer;
                                 save_state=true,
                                 burn_in = 0,
                                 progress=PROGRESS[],
                                 progressname="Chain 1",
                                 callback=nothing,
                                 thinning=1,
                                 kwargs...)

    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    Ntotal = thinning * (N - 1) + burn_in + 1

    # Start the timer
    start = time()
    local state
    # Obtain the initial sample and state.
    state, sample = init

    @AbstractMCMC.ifwithprogresslogger progress name = progressname begin
        # Determine threshold values for progress logging
        # (one update per 0.5% of progress)
        if progress
            threshold = Ntotal ÷ 200
            next_update = threshold
        end

        # Discard burn in.
        for i in 1:burn_in
            # Update the progress bar.
            if progress && i >= next_update
                AbstractMCMC.ProgressLogging.@logprogress i / Ntotal
                next_update = i + threshold
            end

            # Obtain the next sample and state.
            state, sample = AbstractMCMC.step(sampler, target, state; kwargs...)
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
        for i in 2:N
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
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
            callback === nothing || callback(rng, target, sampler, sample, state, i; kwargs...)

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
    save_state=save_state,
    stats=stats,
    burn_in=burn_in,
    thinning=thinning,
    kwargs...)
end

function AbstractMCMC.bundle_samples(
    samples::Vector,
    target::Target, #::AbstractMCMC.AbstractModel,
    sampler::Sampler, #::AbstractMCMC.AbstractSampler,
    state;
    save_state = true,
    stats = missing,
    burn_in = 0,
    thinning = 1,
    kwargs...)

    param_names = target.vsyms
    internal_names = [:E, :logp]
    names = [param_names; internal_names]

    # Set up the info tuple.
    if save_state
        info = (target=target, sampler=sampler,
                init=(state, samples[end]))
    else
        info = NamedTuple()
    end

    # Merge in the timing info, if available
    if !ismissing(stats)
        info = merge(info, (start_time=stats.start, stop_time=stats.stop))
    end

    # Conretize the array before giving it to MCMCChains.
    samples = MCMCChains.concretize(samples)

    # Chain construction.
    chain = MCMCChains.Chains(
        samples,
        names,
        (internals = internal_names,);
        info=info,
        start=burn_in + 1,
        thin=thinning)

    return chain
end

##### Naive parallelization #####
#################################

chainsstack(c::AbstractVector{MCMCChains.Chains}) = reduce(chainscat, c)

function AbstractMCMC.sample(model::DynamicPPL.Model,
                             sampler::AbstractMCMC.AbstractSampler,
                             ::MCMCThreads,
                             N::Integer,
                             nchains::Integer;
                             progress=PROGRESS[],
                             progressname="Sampling",
                             resume_from=nothing,
                             kwargs...)

    if resume_from === nothing
        target = TuringTarget(model)
        init = Init(sampler, target; kwargs...)

        if nchains < Threads.nthreads()
            @info string("number of chains: ",
                         nchains,
                         " smaller than number of threads: ",
                         Threads.nthreads(),  ".",
                         " Increase the number of chains to make full use of your threads.")
        end
        if nchains > Threads.nthreads()
            @info string("number of chains: ",
                         nchains,
                         " requesteed larger than number of threads: ",
                         Threads.nthreads(),  ".")
        end

        interval = 1:nchains
        chains = Vector{MCMCChains.Chains}(undef, nchains)
        targets = [deepcopy(target) for _ in interval]
        samplers = [deepcopy(sampler) for _ in interval]
        inits = [Init(sampler, target) for _ in interval]

        Threads.@threads for i in interval
            _sampler = samplers[i]
            _target = targets[i]
            _init = inits[i]
            state, sample = _init
            tune_hyperparameters(_sampler, _target, state; kwargs...)
        end

    else
        @info "Starting from previous run"
        nchains = length(resume_from)
        interval = 1:nchains
        chains = Vector{MCMCChains.Chains}(undef, nchains)
        targets = [chain.info[:target] for chain in resume_from]
        samplers = [chain.info[:sampler] for chain in resume_from]
        inits = [chain.info[:init] for chain in resume_from]
    end

    @AbstractMCMC.ifwithprogresslogger progress name = progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Channel{Bool}(length(interval))
        end

        Threads.@threads for i in interval
            _sampler = samplers[i]
            _target = targets[i]
            _init = inits[i]
            chains[i] = AbstractMCMC.mcmcsample(
                _target,
                _sampler,
                _init,
                N;
                progress=PROGRESS[],
                progressname=string("chain ", i),
                kwargs...)

            # Update the progress bar.
            progress && put!(channel, true)
        end
    end

    return chains
end
