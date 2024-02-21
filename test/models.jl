@testset "Models" begin
    @testset "Rosembrok" begin
        ##################
        ### Rosembrock ### 
        ##################
        T = Float32
        d=2
        a=T(1.0)
        b=T(10.0)
        target = RosenbrockTarget(a, b, d)
        spl = MCHMC(10_000, 0.01;
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
        @test m2 ≈ 1.97 atol = 0.2
        @test s2 ≈ 2.40 atol = 0.5
    end
end
