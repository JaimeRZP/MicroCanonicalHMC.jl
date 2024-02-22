struct Hamiltonian
    ℓπx::Any
    ∂lπ∂x::Any
end

function Hamiltonian(ℓπ, ∂lπ∂θ, inv_transform)
    ℓπx(x) = ℓπ(inv_transform(x))
    ∂lπ∂x(x) = ∂lπ∂θ(inv_transform(x))
    return Hamiltonian(ℓπx, ∂lπ∂x)
end
