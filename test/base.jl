@testset "Types" begin
    T = Float32
    d = 2
    target_default = RosenbrockTarget(1.0, 10.0, d);
    target_T = RosenbrockTarget(T(1.0), T(10.0), d);

    @test eltype(target_default.θ_start) == Float64
    @test eltype(target_T.θ_start) == T
    @test eltype(target_default.h.ℓπx(target_default.θ_start)) == Float64
    @test eltype(target_T.h.ℓπx(target_T.θ_start)) == T

    spl = MCHMC(0, 0.01)
    @test eltype(spl.hyperparameters.eps) == Float64
    t, s = Step(spl, target_T.h, target_T.θ_start)
    @test eltype(s.x) == T

    aspl = MCHMC(0, 0.01; T=T, adaptive=true)
    @test eltype(aspl.hyperparameters.eps) == Float64 
    t, s = Step(aspl, target_T.h, target_T.θ_start)
    @test eltype(aspl.hyperparameters.eps) == Float32 
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
    u = MicroCanonicalHMC.Random_unit_vector(rng, ones(d))
    @test length(u) == d
    @test isapprox(norm(u), 1.0, rtol = 0.0000001)

    p = MicroCanonicalHMC.Partially_refresh_momentum(rng, 0.1, u)
    @test length(p) == d
    @test isapprox(norm(p), 1.0, rtol = 0.0000001)
end

@testset "Init" begin
    d = 10
    m = zeros(d)
    rng = MersenneTwister(1234)
    a=1.0
    b=10.0
    target = RosenbrockTarget(a, b, d)
    spl = MCHMC(0, 0.001)
    _, init = MicroCanonicalHMC.Step(rng, spl, target.h, m)
    @test init.x == m
    m[1:5] .= -1.0
    @test init.g == m
    @test init.dE == 0
    @test init.Feps == 0.0	
    @test init.Weps == 1.0e-5
end

@testset "Step" begin
    d = 2
    rng = MersenneTwister(1234)
    a=1.0
    b=10.0
    target = RosenbrockTarget(a, b, d)
    spl = MCHMC(0, 0.001; eps = 0.01, L = 0.1, sigma = ones(d))
    aspl = MCHMC(0, 0.001; eps = 0.01, L = 0.1, sigma = ones(d), adaptive = true)
    _, init = MicroCanonicalHMC.Step(rng, spl, target.h, target.θ_start)
    _, step = MicroCanonicalHMC.Step(rng, spl, init)
    _, astep = MicroCanonicalHMC.Step(rng, aspl, init)
    @test spl.hyperparameters.eps == 0.01
    @test aspl.hyperparameters.eps != 0.01
    @test step.x == astep.x
end

@testset "Transformations" begin
    d = 4
    rng = MersenneTwister(1234)
    a=1.0
    b=10.0       
    Λ=10.0
    transform(θ) = Λ .* θ
    inv_transform(x) = x ./ Λ
    target = RosenbrockTarget(a, b, d;
        transform = transform,
        inv_transform = inv_transform)
    spl = MCHMC(0, 0.001; eps = 0.01, L = 0.1, sigma = ones(d))

    x_start = target.transform(target.θ_start)
    @test x_start == 10 .* target.θ_start

    t, s = MicroCanonicalHMC.Step(rng, spl, target.h, x_start;
        inv_transform=target.inv_transform)
    @test t.θ ≈ target.inv_transform(s.x) atol = 0.0001
    @test t.θ ≈ target.θ_start atol = 0.0001

    samples = Sample(spl, target, 2; include_latent=true)
    θ = samples[1:d, 1][:]
    x = samples[d+1:2d, 1][:]
    @test θ ≈ target.θ_start atol = 0.0001
    @test x ≈ x_start atol = 0.0001
    θ = samples[1:d, 2][:]
    x = samples[d+1:2d, 2][:]
    @test θ ≈ target.inv_transform(x) atol = 0.0001
end
