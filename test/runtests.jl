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

        S = 4
        Q = zeros(S,S)
        Q[end,:] .= 1/(S-1)
        Q[end,end] = -1
        p1 = UniformUnmasking()
        p2 = GeneralDiscrete(Q)
        X0 = CategoricalLikelihood(rand(S,100))
        for f in [forward, backward]
            XtF1 = f(X0, p1, 0.234)
            XtF2 = f(X0, p2, 0.234)
            @test isapprox(XtF1.dist, XtF2.dist)
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

    @testset "HPiQ" begin

        tree = PiNode(1.0)
        child1 = PiNode(2.0)
        child2 = PiNode(3.0)
        add_child!(tree, child1)
        add_child!(tree, child2)
        add_child!(child1, PiLeaf(1))
        add_child!(child1, PiLeaf(2))
        add_child!(child2, PiLeaf(3))
        add_child!(child2, PiLeaf(4))
        init_leaf_indices!(tree)

        π = [0.1, 0.4, 0.3, 0.2]
        p_hpiq = HPiQ(tree, π)
        N = length(π)
        @testset "HPiQ create Q" begin
            Q = [-2.5; 0.5; 0.1; 0.1;; 2.0; -1.0; 0.4; 0.4;; 0.3;  0.3; -1.9; 2.1;; 0.2; 0.2; 1.4; -2.6;;]
            @test isapprox(Q, ForwardBackward.HPiQ_Qmatrix(p_hpiq), atol=1e-9)
        end
        @testset "HPiQ Temporal Consistency" begin
            for f in [backward, forward]
                X0 = CategoricalLikelihood(rand(N, 5, 6))
                t1 = 0.123
                t2 = 0.234 .* rand(5, 6) 
                
                X_step1 = f(X0, p_hpiq, t1)
                X_step2 = f(X_step1, p_hpiq, t2)

                t_hop = t1 .+ t2
                X_hop = f(X0, p_hpiq, t_hop)
                @test isapprox(X_hop.dist, X_step2.dist, atol=1e-9)
                @test isapprox(X_hop.log_norm_const, X_step2.log_norm_const, atol=1e-9)
            end
        end

        @testset "HPiQ Equivalence with GeneralDiscrete" begin

            Q = ForwardBackward.HPiQ_Qmatrix(p_hpiq)
            p_general = GeneralDiscrete(Q)
            
            X0 = CategoricalLikelihood(rand(N, 10)) 
            dt = 0.456 

            for f in [forward, backward]

                Xt_hpiq = f(X0, p_hpiq, dt)
                Xt_general = f(X0, p_general, dt)

                @test isapprox(Xt_hpiq.dist, Xt_general.dist, atol=1e-9)
                @test isapprox(Xt_hpiq.log_norm_const, Xt_general.log_norm_const, atol=1e-9)
            end
        end

    end
end
