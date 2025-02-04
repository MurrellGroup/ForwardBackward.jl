using ForwardBackward
using Test

@testset "ForwardBackward.jl" begin
    @testset "Temporal Consistency - Continuous" begin
        for p in [OrnsteinUhlenbeck(), BrownianMotion()]
            for f in [backward, forward]
                X0 = rand(GaussianLikelihood(randn(3,4,5), rand(3,4,5), rand(3,4,5)))
                XtF = f(X0, p, 0.0)
                XtF2 = f(XtF, p, 0.123)
                XtF3 = f(XtF2, p, 0.234 .* ones(5))
                XtHop = f(X0, p, 0.123 .+ 0.234 .* ones(3,4,5))
                @test isapprox(XtHop.mu, XtF3.mu)
                @test isapprox(XtHop.var, XtF3.var)
                @test isapprox(XtHop.log_norm_const, XtF3.log_norm_const)
            end
        end
    end

    @testset "Temporal Consistency - Discrete" begin
        for p in [PiQ(sumnorm(rand(20)))]
            for f in [backward, forward]
                X0 = DiscreteState(20, rand(1:20,100,4))
                XtF = f(X0, p, 0.0)
                XtF2 = f(XtF, p, 0.123)
                XtF3 = f(XtF2, p, 0.234 .* ones(4))
                XtHop = f(X0, p, 0.123 .+ 0.234 .* ones(100,4))
                @test isapprox(XtHop.dist, XtF3.dist)
                @test isapprox(XtHop.log_norm_const, XtF3.log_norm_const)
            end
        end
    end

    @testset "Some discrete equivalences" begin
        Q = ones(4,4)
        for i in 1:4
            Q[i,i] = -3
        end
        Q ./= 3 #To make the expected subs per unit time = 1
        p1 = PiQ(sumnorm(ones(4) ./ 4), normalize = true)
        p2 = UniformDiscrete()
        p3 = GeneralDiscrete(Q)
        X0 = DiscreteState(4, rand(1:4,100))
        for f in [forward, backward]
            XtF1 = f(X0, p1, 0.234)
            XtF2 = f(X0, p2, 0.234)
            XtF3 = f(X0, p3, 0.234)
            @test isapprox(XtF1.dist, XtF2.dist)
            @test isapprox(XtF1.dist, XtF3.dist)
        end
    end

    @testset "Some continuous equivalences" begin
        p = BrownianMotion()
        X0 = rand(GaussianLikelihood(randn(3,4,5), rand(3,4,5), rand(3,4,5)))
        X1 = rand(GaussianLikelihood(randn(3,4,5), rand(3,4,5), rand(3,4,5)))
        XtF = forward(X0, p, 0.3)
        XtB = backward(X1, p, 0.7)
        Xt = XtF ⊙ XtB
        Xt_I = interpolate(X0, X1, 0.3 .* ones(4,5), 0.7 .* ones(4,5))
        Xt_D = endpoint_conditioned_sample(X0, X1, Deterministic(), 0.3 .* ones(4,5))
        @test isapprox(Xt_D.state, Xt.mu)
        @test isapprox(Xt_I.state, Xt.mu)
    end
end
