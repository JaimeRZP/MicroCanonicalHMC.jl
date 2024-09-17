@testset "Models" begin
    @testset "Init Params" begin
        ##################
        ### Rosembrock ### 
        ##################
        T = Float64
        d=2
        a=T(1.0)
        b=T(1.0)
        Λ=T(3.0)
        init_params = T.([0.1, 0.1])
        transform(θ) = Λ .* θ
        inv_transform(x) = x ./ Λ
        target = RosenbrockTarget(a, b, d;
            transform = transform,
            inv_transform = inv_transform)
        spl = MCHMC(0, 0.01;
        L = sqrt(2), sigma = ones(target.d),
        tune_L = false, tune_sigma = false, adaptive = true)
        samples = Sample(spl, target, 2; 
            init_params = init_params)
        θ_1 = samples[:, 1][1:2]
        println(θ_1)
        @test θ_1 ≈ init_params atol = transform(0.000000001)
    end

    @testset "Rosembrok" begin
        ##################
        ### Rosembrock ### 
        ##################
        T = Float32
        d=2
        a=T(1.0)
        b=T(10.0)
        target = RosenbrockTarget(a, b, d)
        spl = MCHMC(50_000, 0.01;
        L = sqrt(2), sigma = ones(target.d),
        tune_L = false, tune_sigma = false, adaptive = true)
        samples = Sample(spl, target, 200_000; dialog = true)
        d1 = samples[1, :]
        d2 = samples[2, :]
        mm1, m1, s1, = (median(d1), mean(d1), std(d1))
        mm2, m2, s2, = (median(d2), mean(d2), std(d2))
        @test eltype(samples[1]) == T
        @test mm1 ≈ 1.00 atol = 0.2
        @test m1 ≈ 1.00 atol = 0.2
        @test s1 ≈ 1.00 atol = 0.3
        @test mm2 ≈ 1.13 atol = 0.2
        @test m2 ≈ 2.22 atol = 0.2
        @test s2 ≈ 2.40 atol = 0.5
    end

    @testset "Rosembrok Transformed" begin
        ##################
        ### Rosembrock ### 
        ##################
        T = Float64
        d=2
        a=T(1.0)
        b=T(1.0)
        Λ=T(3.0)
        transform(θ) = Λ .* θ
        inv_transform(x) = x ./ Λ
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

        xx1 = transform(θ1)
        xx2 = transform(θ2)

        mx1, sx1, = (mean(x1), var(x1))
        mx2, sx2, = (mean(x2), var(x2))
        mxx1, sxx1, = (mean(xx1), var(xx1))
        mxx2, sxx2, = (mean(xx2), var(xx2))

        @test mx1 ≈ mxx1 atol = transform(0.1)
        @test sx1 ≈ sxx1 atol = transform(0.1)
        @test mx2 ≈ mxx2 atol = transform(0.1)
        @test sx2 ≈ sxx2 atol = transform(0.1)
    end
end
