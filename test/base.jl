@testset "Types" begin
    target_default = RosenbrockTarget(1.0, 10.0; d=2);
    T = Float32
    target_T = RosenbrockTarget(1.0, 10.0; T=T, d=2);

    @test eltype(target_default.θ_start) == Float64
    @test eltype(target_T.θ_start) == T
    @test eltype(target_default.h.ℓπ(target_default.θ_start)) == Float64
    @test eltype(target_T.h.ℓπ(target_T.θ_start)) == T

    spl = MCHMC(0, 0.01)
    _, _, _, = MicroCanonicalHMC.tune_what(spl, 10; T=Float32)
    #println(spl.hyperparameters)
    #@test eltype(spl.hyperparameters.eps) == Float32
    #@test eltype(spl.hyperparameters.L) == Float32
    #@test eltype(spl.hyperparameters.sigma) == Float32
    t, s = Step(spl, target_T.h, target_T.θ_start)
    @test eltype(s.x) == T
end
@testset "Settings" begin
    spl = MCHMC(
        10_000,
        0.1;
        integrator = "MN",
        eps = 0.1,
        L = 0.1,
        sigma = [1.0],
        gamma = 2.0,
        sigma_xi = 2.0,
    )

    hp = spl.hyperparameters
    dy = spl.hamiltonian_dynamics

    @test spl.TEV == 0.1
    @test spl.nadapt == 10_000
    @test spl.hamiltonian_dynamics == MicroCanonicalHMC.Minimal_norm

    @test hp.eps == 0.1
    @test hp.L == 0.1
    @test hp.sigma == [1.0]
    @test hp.gamma == 2.0
    @test hp.sigma_xi == 2.0
 
end

@testset "Partially_refresh_momentum" begin
    d = 10
    rng = MersenneTwister(0)
    u = MicroCanonicalHMC.Random_unit_vector(rng, d)
    @test length(u) == d
    @test isapprox(norm(u), 1.0, rtol = 0.0000001)

    p = MicroCanonicalHMC.Partially_refresh_momentum(rng, 0.1, u)
    @test length(p) == d
    @test isapprox(norm(p), 1.0, rtol = 0.0000001)
end

@testset "Init" begin
    d = 10
    m = zeros(d)
    s = Diagonal(ones(d))
    rng = MersenneTwister(1234)
    target = GaussianTarget(m, s)
    spl = MCHMC(0, 0.001)
    _, init = MicroCanonicalHMC.Step(rng, spl, target.h, m)
    @test init.x == m
    @test init.g == m
    @test init.dE == init.Feps == 0.0
    @test init.Weps == 1.0e-5
end

@testset "Step" begin
    d = 10
    m = zeros(d)
    s = Diagonal(ones(d))
    rng = MersenneTwister(1234)
    target = GaussianTarget(m, s)
    spl = MCHMC(0, 0.001; eps = 0.01, L = 0.1, sigma = ones(d))
    aspl = MCHMC(0, 0.001; eps = 0.01, L = 0.1, sigma = ones(d), adaptive = true)
    _, init = MicroCanonicalHMC.Step(rng, spl, target.h, target.θ_start)
    tune_sigma, tune_eps, tune_L = MicroCanonicalHMC.tune_what(spl, d)
    tune_sigma, tune_eps, tune_L = MicroCanonicalHMC.tune_what(aspl, d)
    @test tune_sigma == tune_eps == tune_L == false
    _, step = MicroCanonicalHMC.Step(rng, spl, init)
    _, astep = MicroCanonicalHMC.Step(rng, aspl, init)
    @test spl.hyperparameters.eps == 0.01
    @test aspl.hyperparameters.eps != 0.01
    @test step.x == astep.x
end
