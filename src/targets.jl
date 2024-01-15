NoTransform(x) = x

mutable struct Target
    d::Int
    vsyms::Any
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

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

CustomTarget(nlogp, grad_nlogp, priors; kwargs...) = begin
    d = length(priors)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    function prior_draw()
        x = [rand(dist) for dist in priors]
        xt = transform(x)
        return xt
    end
    hamiltonian = Hamiltonian(nlogp, grad_nlogp)
    Target(d, hamiltonian, NoTransform, NoTransform, prior_draw)
end

GaussianTarget(_mean::AbstractVector, _cov::AbstractMatrix) = begin
    d = length(_mean)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    _gaussian = MvNormal(_mean, _cov)
    ℓπ(θ::AbstractVector) = logpdf(_gaussian, θ)
    ∂lπ∂θ(θ::AbstractVector) = (logpdf(_gaussian, θ), gradlogpdf(_gaussian, θ))
    hamiltonian = Hamiltonian(ℓπ, ∂lπ∂θ)

    function prior_draw()
        xt = rand(MvNormal(zeros(d), ones(d)))
        return xt
    end

    Target(d, vsyms, hamiltonian, NoTransform, NoTransform, prior_draw)
end

RosenbrockTarget(a, b; kwargs...) = begin
    kwargs = Dict(kwargs)
    d = kwargs[:d]
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    function ℓπ(x; a = a, b = b)
        x1 = x[1:Int(d / 2)]
        x2 = x[Int(d / 2)+1:end]
        m = @.((a - x1)^2 + b * (x2 - x1^2)^2)
        return -0.5 * sum(m)
    end

    function ∂lπ∂θ(x)
        return ℓπ(x), ForwardDiff.gradient(ℓπ, x)
    end

    hamiltonian = Hamiltonian(ℓπ, ∂lπ∂θ)

    function prior_draw()
        x = rand(MvNormal(zeros(d), ones(d)))
        return x
    end

    Target(d, vsyms, hamiltonian, NoTransform, NoTransform, prior_draw)
end
