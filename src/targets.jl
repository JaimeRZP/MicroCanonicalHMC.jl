mutable struct TuringTarget <: Target
    model::DynamicPPL.Model
    d::Int
    vsyms
    dists
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    hess_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
    MAP::AbstractVector
    MAP_t::AbstractVector
end

function _get_dists(vi)
    mds = values(vi.metadata)
    return [md.dists[1] for md in mds]
end

function _name_variables(vi, dist_lengths)
    vsyms = keys(vi)
    names = []
    for (vsym, dist_length) in zip(vsyms, dist_lengths)
        if dist_length==1
            name = [vsym]
            append!(names, name)
        else
            name = [DynamicPPL.VarName(Symbol(vsym, i,)) for i in 1:dist_length]
            append!(names, name)
         end
    end
    return names
end

TuringTarget(model; compute_MAP=false, kwargs...) = begin
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

    function nlogp(xt)
        return -ℓπ(xt)
    end

    function grad_nlogp(xt)
        return ForwardDiff.gradient(nlogp, xt)
    end

    function nlogp_grad_nlogp(xt)
        return -1 .* ∂lπ∂θ(xt)
    end

    function hess_nlogp(xt)
        return ForwardDiff.hessian(nlogp, xt)
    end

    function prior_draw(key)
        ctxt = model.context
        vi = DynamicPPL.VarInfo(model, ctxt)
        vi_t = Turing.link!!(vi, model)
        return vi_t[DynamicPPL.SampleFromPrior()]
    end

    if compute_MAP
        MAP_t = Optim.minimizer(optimize(nlogp, prior_draw(0.0), Newton(); autodiff = :forward))
        MAP = inv_transform(MAP_t)
    else
        MAP = MAP_t = zeros(d)
    end

    TuringTarget(
               model,
               d,
               vsyms,
               dists,
               nlogp,
               grad_nlogp,
               nlogp_grad_nlogp,
               hess_nlogp,
               transform,
               inv_transform,
               prior_draw,
               MAP,
               MAP_t)
end


mutable struct CustomTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

CustomTarget(nlogp, grad_nlogp, priors; kwargs...) = begin
    d = length(priors)

    #function transform(xs)
    #    xxs = [Bijectors.invlink(dist, x) for (dist, x) in zip(priors, xs)]
    #    return xxs
    #end

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function prior_draw(key)
        x = [rand(key, dist) for dist in priors]
        xt = transform(x)
        return xt
    end

    CustomTarget(d,
               nlogp,
               grad_nlogp,
               transform,
               inv_transform,
               prior_draw)
end

mutable struct GaussianTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

GaussianTarget(_mean::AbstractVector ,_cov::AbstractMatrix) = begin
    d = length(_mean)
    _gaussian = MvNormal(_mean, _cov)
    ℓπ(θ::AbstractVector) = logpdf(_gaussian, θ)
    ∂lπ∂θ(θ::AbstractVector) = gradlogpdf(_gaussian, θ)

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function nlogp(x)
        xt = transform(x)
        return -ℓπ(xt)
    end

    function grad_nlogp(x)
        xt = transform(x)
        return -∂lπ∂θ(xt)
    end

    function nlogp_grad_nlogp(x)
        l = nlogp(x)
        g = grad_nlogp(x)
        return l, g
    end

    function prior_draw(key)
        xt = rand(MvNormal(zeros(d), ones(d)))
        return xt
    end

    GaussianTarget(d,
    nlogp,
    grad_nlogp,
    nlogp_grad_nlogp,
    transform,
    inv_transform,
    prior_draw)
end

mutable struct RosenbrockTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

struct Rosenbrock{Tμ,Ta,Tb}
    μ::Tμ #Off diag size
    a::Ta
    b::Tb
end

RosenbrockTarget(Tμ, Ta, Tb; kwargs...) = begin
    kwargs = Dict(kwargs)
    D = Rosenbrock(Tμ, Ta, Tb)
    d = kwargs[:d]

    block = [1.0 D.μ; D.μ 1.0]
    _cov = BlockDiagonal([block for _ in 1:(d/2)])

    function _mean(θ::AbstractVector)
        i = floor.(Int,(1:(d/2)))
        even_i = floor.(Int, 2 .* i)
        odd_i = floor.(Int, 2 .* i .- 1)
        u = ones(d)
        u[even_i] .= θ[even_i] ./ D.a
        u[odd_i] .= D.a .* (θ[odd_i] .- D.b .* (θ[even_i] .^ 2 .+ D.a ^ 2))
        return u
    end

    rosenbrock(θ::AbstractVector) = MvNormal(_mean(θ), _cov)
    ℓπ(θ::AbstractVector) = logpdf(rosenbrock(θ), θ)
    ∂lπ∂θ(θ::AbstractVector) = gradlogpdf(rosenbrock(θ), θ)

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function nlogp(x)
        xt = transform(x)
        return -ℓπ(xt)
    end

    function grad_nlogp(x)
        xt = transform(x)
        return -∂lπ∂θ(xt)
    end

    function nlogp_grad_nlogp(x)
        l = nlogp(x)
        g = grad_nlogp(x)
        return l, g
    end

    function prior_draw(key)
        xt = rand(MvNormal(zeros(d), ones(d)))
        return xt
    end

    RosenbrockTarget(d,
    nlogp,
    grad_nlogp,
    nlogp_grad_nlogp,
    transform,
    inv_transform,
    prior_draw)
end

mutable struct NealFunnelTarget <: Target
    model::DynamicPPL.Model
    d::Int
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

NealFunnelTarget(model; d=0, kwargs...) = begin
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)

    ℓ = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(vi, model, ctxt))
    ℓπ(x) = LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)

    function nlogp(xt)
        return -ℓπ(xt)
    end

    function grad_nlogp(xt)
        return ForwardDiff.gradient(nlogp, xt)
    end

    function nlogp_grad_nlogp(xt)
        return -1 .* ∂lπ∂θ(xt)
    end

    function prior_draw(key)
        ctxt = model.context
        vi = DynamicPPL.VarInfo(model, ctxt)
        return vi[DynamicPPL.SampleFromPrior()]
    end

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    NealFunnelTarget(
               model,
               d,
               nlogp,
               grad_nlogp,
               nlogp_grad_nlogp,
               transform,
               inv_transform,
               prior_draw)
end