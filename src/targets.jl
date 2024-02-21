NoTransform(x) = x

mutable struct Target{T}
    d::Int
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    θ_start::Vector{T}
    θ_names::Vector{String}
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

function GaussianTarget(d::Int; T::Type=Float64)
    m = zeros(T, d)
    s = Diagonal(ones(T, d))
    return GaussianTarget(m, s; kwargs...)
end

function GaussianTarget(_mean::Vector{T}, _cov::AbstractMatrix{T}) where {T}
    d = length(_mean)
    _gaussian = MvNormal(_mean, _cov)
    function ℓπ(θ::Vector{T}) where {T}
        return T(logpdf(_gaussian, θ))
    end
    function ∂lπ∂θ(θ::Vector{T}) where {T}
        return (T(logpdf(_gaussian, θ)), T.(gradlogpdf(_gaussian, θ)))
    end
    θ_start = T.(rand(MvNormal(zeros(d), ones(d))))
    return CustomTarget(ℓπ, ∂lπ∂θ, θ_start)
end

function RosenbrockTarget(a::T, b::T, d::Int) where{T}
    function ℓπ(x::Vector{T}; a = a, b = b) where {T}
        a = T(a)
        b = T(b)
        x1 = x[1:Int(d / 2)]
        x2 = x[Int(d / 2)+1:end]
        m = @.((a - x1)^2 + b * (x2 - x1^2)^2)
        return -T(1/2) * sum(m)
    end
    function ∂lπ∂θ(x::Vector{T}) where {T}
        return ℓπ(x), ForwardDiff.gradient(ℓπ, x)
    end
    θ_start = T.(rand(MvNormal(zeros(d), ones(d))))
    return CustomTarget(ℓπ, ∂lπ∂θ, θ_start)
end
