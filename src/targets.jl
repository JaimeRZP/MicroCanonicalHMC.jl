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
    return Target(d, Hamiltonian(nlogp, grad_nlogp, inv_transform), transform, inv_transform, θ_start, θ_names)
end

function RosenbrockTarget(a::T, b::T, d::Int; kwargs...) where{T}
    function ℓπ(θ::AbstractVector{T}; a = a, b = b) where {T}
        a = T(a)
        b = T(b)
        θ1 = θ[1:Int(d / 2)]
        θ2 = θ[Int(d / 2)+1:end]
        m = @.((a - θ1)^2 + b * (θ2 - θ1^2)^2)
        return -T(1/2) * sum(m)
    end
    function ∂lπ∂θ(θ::AbstractVector) 
        return ℓπ(θ), ForwardDiff.gradient(ℓπ, θ)
    end
    θ_start = T.(rand(MvNormal(zeros(d), ones(d))))
    return CustomTarget(ℓπ, ∂lπ∂θ, θ_start; kwargs...)
end
