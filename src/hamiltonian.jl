struct Hamiltonian
    ℓπ::Any
    ∂lπ∂θ::Any
end

function Hamiltonian(ℓ)
    ℓπ(x) = LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)
    return Hamiltonian(ℓπ, ∂lπ∂θ)
end
