NoTransform(x) = x

mutable struct Target{T}
    d::Int
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    θ_start::AbstractVector{T}
    θ_names::AbstractVector{String}
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

function RosenbrockTarget(a::T, b::T, d::Int) where{T}
    function ℓπ(θ::AbstractVector{T}; a = a, b = b) where {T}
        a = T(a)
        b = T(b)
        θ1 = θ[1:Int(d / 2)]
        θ2 = θ[Int(d / 2)+1:end]
        m = @.((a - θ1)^2 + b * (θ2 - θ1^2)^2)
        return -T(1/2) * sum(m)
    end
    function ∂lπ∂θ(θ::AbstractVector{T}) where {T}
        return ℓπ(θ), ForwardDiff.gradient(ℓπ, θ)
    end
    θ_start = T.(rand(MvNormal(zeros(d), ones(d))))
    return CustomTarget(ℓπ, ∂lπ∂θ, θ_start)
end
