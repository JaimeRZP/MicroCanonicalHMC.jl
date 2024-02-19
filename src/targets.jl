#=
mutable struct TuringTarget <: Target
    model::DynamicPPL.Model
    d::Int
    vsyms::Any
    dists::Any
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

function _get_dists(vi)
    mds = values(vi.metadata)
    return [md.dists[1] for md in mds]
end

function _name_variables(vi, dist_lengths)
    vsyms = keys(vi)
    names = []
    for (vsym, dist_length) in zip(vsyms, dist_lengths)
        if dist_length == 1
            name = [vsym]
            append!(names, name)
        else
            name = [DynamicPPL.VarName(Symbol(vsym, i)) for i = 1:dist_length]
            append!(names, name)
        end
    end
    return names
end

TuringTarget(model; kwargs...) = begin
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)
    vi_t = Turing.link!!(vi, model)
    dists = _get_dists(vi)
    dist_lengths = [length(dist) for dist in dists]
    vsyms = _name_variables(vi, dist_lengths)
    d = length(vsyms)

    ℓ = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(vi_t, model, ctxt))
    ℓπ(x) = LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)
    hamiltonian = Hamiltonian(ℓπ, ∂lπ∂θ)

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

    function prior_draw()
        ctxt = model.context
        vi = DynamicPPL.VarInfo(model, ctxt)
        vi_t = Turing.link!!(vi, model)
        return vi_t[DynamicPPL.SampleFromPrior()]
    end

    TuringTarget(model, d, vsyms, dists, hamiltonian, transform, inv_transform, prior_draw)
end
=#

NoTransform(x) = x

mutable struct Target
    T::Type
    d::Int
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    θ_start::Vector{Float64}
    θ_names::Vector{String}
end

function CustomTarget(nlogp, grad_nlogp, θ_start::Vector{Float64};
    T::Type=Float64,
    θ_names=nothing,
    transform=NoTransform,
    inv_transform=NoTransform) 
    d = length(θ_start)
    if θ_names==nothing
        θ_names = [string("θ_", i) for i=1:d]
    end
    return Target(T, d, Hamiltonian(nlogp, grad_nlogp), transform, inv_transform, θ_start, θ_names)
end

function GaussianTarget(_mean::AbstractVector, _cov::AbstractMatrix;
    T::Type=Float64)
    d = length(_mean)
    _gaussian = MvNormal(_mean, _cov)
    ℓπ(θ::AbstractVector) = logpdf(_gaussian, θ)
    ∂lπ∂θ(θ::AbstractVector) = (logpdf(_gaussian, θ), gradlogpdf(_gaussian, θ))
    θ_start = rand(MvNormal(zeros(d), ones(d)))
    return CustomTarget(ℓπ, ∂lπ∂θ, θ_start; T=T)
end

function RosenbrockTarget(a, b; T::Type=Float64, kwargs...)
    kwargs = Dict(kwargs)
    d = kwargs[:d]
    function ℓπ(x; a = a, b = b)
        x1 = x[1:Int(d / 2)]
        x2 = x[Int(d / 2)+1:end]
        m = @.((a - x1)^2 + b * (x2 - x1^2)^2)
        return -(1/2) * sum(m)
    end
    function ∂lπ∂θ(x)
        return ℓπ(x), ForwardDiff.gradient(ℓπ, x)
    end
    θ_start = rand(MvNormal(zeros(d), ones(d)))
    return CustomTarget(ℓπ, ∂lπ∂θ, θ_start; T=T)
end
