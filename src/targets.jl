NoTransform(x) = x

mutable struct Target{T}
    d::Int
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    θ_start::AbstractVector{T}
    θ_names::AbstractVector{String}
end

function CustomTarget(nlog, θ_start::AbstractVector; kwargs...)
    function grad_nlogp(θ::AbstractVector)
        return nlog(θ), ForwardDiff.gradient(nlog, θ)
    end
    return CustomTarget(nlog, grad_nlogp, θ_start; kwargs...)
    
end

function CustomTarget(nlogp, grad_nlogp, θ_start::AbstractVector;
    θ_names=nothing,
    transform=NoTransform,
    inv_transform=NoTransform) 
    d = length(θ_start)
    if θ_names==nothing
        θ_names = [string("θ_", i) for i=1:d]
    end
    return Target(d, Hamiltonian(nlogp, grad_nlogp), transform, inv_transform, θ_start, θ_names)
end

function RosenbrockTarget(a::T, b::T, d::Int;
    transform=NoTransform,
    inv_transform=NoTransform,
    kwargs...) where{T}
    function ℓπ(x::AbstractVector{T}; a = a, b = b) where {T}
        θ = inv_transform(x)
        a = T(a)
        b = T(b)
        θ1 = θ[1:Int(d / 2)]
        θ2 = θ[Int(d / 2)+1:end]
        m = @.((a - θ1)^2 + b * (θ2 - θ1^2)^2)
        return -T(1/2) * sum(m)
    end
    function ∂lπ∂x(x::AbstractVector) 
        return ℓπ(x), ForwardDiff.gradient(ℓπ, x)
    end
    θ_start = T.(rand(MvNormal(zeros(d), ones(d))))
    return CustomTarget(
        ℓπ,
        ∂lπ∂x,
        θ_start;
        transform=transform,
        inv_transform=inv_transform,
        kwargs...)
end

function TuringTarget(model; kwargs...)
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)
    vi_t = Turing.link!!(vi, model)
    θ_start = vi[DynamicPPL.SampleFromPrior()]
    dists = _get_dists(vi)
    dist_lengths = [length(dist) for dist in dists]
    θ_names = _name_variables(vi, dist_lengths)
    d = length(θ_names)
    ℓ = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(vi_t, model, ctxt))
    ℓπ(x) = LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)
    
    function _reshape_params(x::AbstractVector)
        xx = []
        idx = 0
        for dist_length in dist_lengths
            append!(xx, [x[idx+1:idx+dist_length]])
            idx += dist_length
        end
        return xx
    end

    function transform(x)
        x = _reshape_params(x)
        xt = [Bijectors.link(dist, par) for (dist, par) in zip(dists, x)]
        return vcat(xt...)
    end

    function inv_transform(xt)
        xt = _reshape_params(xt)
        x = [Bijectors.invlink(dist, par) for (dist, par) in zip(dists, xt)]
        return vcat(x...)
    end

    return CustomTarget(ℓπ, ∂lπ∂θ, θ_start;
        transform=transform, 
        inv_transform=inv_transform, 
        θ_names=θ_names)
end
