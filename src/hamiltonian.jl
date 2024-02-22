struct Hamiltonian
    ℓπx::Any
    ∂lπ∂x::Any
end

function Hamiltonian(ℓ)
    ℓπ(x) = LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂x(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)
    return Hamiltonian(ℓπ, ∂lπ∂x)
end