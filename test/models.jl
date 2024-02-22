@testset "Models" begin
    @testset "Rosembrok" begin
        ##################
        ### Rosembrock ### 
        ##################
        T = Float32
        d=2
        a=T(1.0)
        b=T(1.0)
        target = RosenbrockTarget(a, b, d)
        spl = MCHMC(10_000, 0.01;
        L = sqrt(2), sigma = ones(target.d),
        tune_L = false, tune_sigma = false, adaptive = true)
        samples = Sample(spl, target, 200_000)
        d1 = samples[1, :]
        d2 = samples[2, :]
        mm1, m1, s1, = (median(d1), mean(d1), var(d1))
        mm2, m2, s2, = (median(d2), mean(d2), var(d2))
        @test eltype(samples[1]) == T
        @test mm1 ≈ 1.00 atol = 0.2
        @test m1 ≈ 1.00 atol = 0.2
        @test s1 ≈ 1.00 atol = 0.3
        @test mm2 ≈ 1.40 atol = 0.2
        @test m2 ≈ 1.97 atol = 0.2
        @test s2 ≈ 6.70 atol = 0.5
    end

    @testset "Rosembrok Transformed" begin
        ##################
        ### Rosembrock ### 
        ##################
        T = Float64
        d=2
        a=T(1.0)
        b=T(1.0)
        Λ=T(10.0)
        inv_transform(θ) = Λ .* θ
        transform(x) = x ./ Λ
        target = RosenbrockTarget(a, b, d;
            transform = transform,
            inv_transform = inv_transform)
        spl = MCHMC(10_000, 0.01;
        L = sqrt(2), sigma = ones(target.d),
        tune_L = false, tune_sigma = false, adaptive = true)
        samples = Sample(spl, target, 200_000; include_latent=true)
        θ1 = samples[1, :]
        θ2 = samples[2, :]
        x1 = samples[3, :]
        x2 = samples[4, :]
        mθ1, sθ1, = (mean(θ1), var(θ1))
        mθ2, sθ2, = (mean(θ2), var(θ2))
        @test mθ1 ≈ 1.00 atol = 0.2
        @test sθ1 ≈ 1.00 atol = 0.3
        @test mθ2 ≈ 1.97 atol = 0.2
        @test sθ2 ≈ 6.70 atol = 0.5
        mx1, sx1, = (mean(x1), var(x1))
        mx2, sx2, = (mean(x2), var(x2))
        @test mx1 ≈ transform(1.00) atol = transform(0.2)
        @test sx1 ≈ transform(1.00) atol = transform(0.3)
        @test mx2 ≈ transform(1.97) atol = transform(0.2)
        @test sx2 ≈ transform(6.70) atol = transform(0.5)
    end
end
