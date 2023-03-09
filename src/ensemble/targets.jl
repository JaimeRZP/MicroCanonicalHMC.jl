mutable struct ParallelTarget <: Target
    target::Target
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

ParallelTarget(target::Target, nchains) = begin
    d = target.d
    function transform(xs)
        xs_t = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            xs_t[i, :] .= target.transform(xs[i, :])
        end
        return xs_t
    end

    function inv_transform(xs_t)
        xs = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            xs[i, :] .= target.inv_transform(xs_t[i, :])
        end
        return xs
    end

    function nlogp(xs_t)
        ls = Vector{Real}(undef, nchains)
        @inbounds Threads.@threads :static for i in 1:nchains
            ls[i] = target.nlogp(xs_t[i, :])
        end
        return ls
    end

    function grad_nlogp(xs_t)
        gs = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            gs[i, :] .= target.grad_nlogp(xs_t[i, :])
        end
        return gs
    end

    function nlogp_grad_nlogp(xs_t)
        ls = Vector{Real}(undef, nchains)
        gs = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            ls[i], = target.nlogp(xs_t[i, :])
            gs[i, :] = target.grad_nlogp(xs_t[i, :])
        end
        return ls , gs
    end

    function prior_draw(key)
        xs_t = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            xs_t[i, :] .= target.prior_draw(key)
        end
        return xs_t
    end

    ParallelTarget(
        target,
        nlogp,
        grad_nlogp,
        nlogp_grad_nlogp,
        transform,
        inv_transform,
        prior_draw)
end