@testset "Models" begin
    @testset "Gaussian" begin
        ################
        ### Gaussian ###
        ################
        d = 20
        k = 100
        m = Vector(LinRange(1, 100, d))
        e = 10 .^ LinRange(log10(1 / sqrt(k)), log10(sqrt(k)), d)
        cov_matt = Diagonal(e)
        target = GaussianTarget(m, cov_matt)

        spl = MCHMC(10_000, 0.01; init_eps = sqrt(d))
        samples_mchmc = Sample(spl, target, 100_000; dialog = true)
        samples_mchmc_adaptive =
            Sample(spl, target, 100_000; adaptive = true, dialog = true)

        _samples_mchmc = mapreduce(permutedims, vcat, samples_mchmc)
        s1 = std(_samples_mchmc, dims = 1)[1:end-3]
        m1 = mean(_samples_mchmc, dims = 1)[1:end-3]

        _samples_mchmc_adaptive = mapreduce(permutedims, vcat, samples_mchmc_adaptive)
        s2 = std(_samples_mchmc_adaptive, dims = 1)[1:end-3]
        m2 = mean(_samples_mchmc_adaptive, dims = 1)[1:end-3]

        @test mean((m1 .- m) ./ sqrt.(e)) ≈ 0.0 atol = 0.2
        @test mean(s1 ./ sqrt.(e) .- 1) ≈ 0.0 atol = 0.2
        @test mean((m2 .- m) ./ sqrt.(e)) ≈ 0.0 atol = 0.2
        @test mean(s2 ./ sqrt.(e) .- 1) ≈ 0.0 atol = 0.2
    end

    @testset "Rosembrok" begin
        ##################
        ### Rosembrock ### 
        ##################
        rng = MersenneTwister(1234)
        target = RosenbrockTarget(1.0, 10.0; d = 2)
        spl = MCHMC(10_000, 0.01; L = sqrt(2), sigma = ones(target.d), adaptive = true)
        samples = Sample(rng, spl, target, 200_000; dialog = true)
        d1 = [sample[1] for sample in samples]
        d2 = [sample[2] for sample in samples]
        mm1, m1, s1, = (median(d1), mean(d1), std(d1))
        mm2, m2, s2, = (median(d2), mean(d2), std(d2))
        @test mm1 ≈ 1.00 atol = 0.2
        @test m1 ≈ 1.00 atol = 0.2
        @test s1 ≈ 1.00 atol = 0.3
        @test mm2 ≈ 1.13 atol = 0.2
        @test m2 ≈ 1.97 atol = 0.2
        @test s2 ≈ 2.40 atol = 0.5
    end
end
