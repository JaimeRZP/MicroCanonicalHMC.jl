@testset "Turing" begin
    @testset "" begin

        d = 10
        @model function funnel()
            θ ~ Truncated(Normal(0, 3), -3, 3)
            z ~ MvNormal(zeros(d - 1), exp(θ) * I)
            return x ~ MvNormal(z, I)
        end

        (; x) = rand(funnel() | (θ=0,))
        model = funnel() | (; x);

        n_adapts = 50 # adaptation steps
        tev = 0.01 # target energy variance
        mchmc = MCHMC(n_adapts, tev; adaptive=true)

        # Sample
        chain = sample(model, externalsampler(mchmc), 100)
        # Just check if it ran
        @test true
    end
end
