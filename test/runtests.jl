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


    @testset "Parameter vectorization - Continuous processes" begin
        # Helper to compute a scalar-by-scalar baseline by looping over parameter grids
        function scalarwise_propagate(f, X0::GaussianLikelihood, mkproc, t)
            dims = size(X0.mu)
            # Expand each parameter over the state grid using broadcasting
            # mkproc should return a NamedTuple of parameters so we can expand them
            params = mkproc()
            expanded = Dict{Symbol, Any}()
            for (k, v) in pairs(params)
                expanded[k] = (zeros(eltype(X0.mu), dims...) .+ v)
            end
            mu_out = similar(X0.mu)
            var_out = similar(X0.var)
            for idx in CartesianIndices(dims)
                # Build scalar process for this site
                if haskey(expanded, :δ)
                    p_scalar = BrownianMotion(expanded[:δ][idx], expanded[:v][idx])
                else
                    p_scalar = OrnsteinUhlenbeck(expanded[:μ][idx], expanded[:v][idx], expanded[:θ][idx])
                end
                X_small = GaussianLikelihood(fill(X0.mu[idx], 1), fill(X0.var[idx], 1), fill(X0.log_norm_const[idx], 1))
                Y_small = f(X_small, p_scalar, t)
                mu_out[idx] = Y_small.mu[1]
                var_out[idx] = Y_small.var[1]
            end
            return mu_out, var_out
        end

        # Common state and time
        X0 = GaussianLikelihood(randn(3,4,5), rand(3,4,5), rand(3,4,5))
        t = 0.234

        # Shapes that broadcast over last and middle dimensions respectively
        δ_last = reshape(randn(5), 1, 1, 5)
        v_mid = reshape(rand(4), 1, 4, 1)
        full_grid = randn(3,4,5)

        @testset "BrownianMotion param shapes" begin
            cases = [
                # (δ, v)
                (0.1, 0.9),
                (δ_last, 0.7),
                (0.2, v_mid),
                (full_grid, 0.3),
                (0.0, full_grid)
            ]
            for (δ, v) in cases
                # Vectorized attempt
                ok = true
                vec_mu = nothing; vec_var = nothing
                try
                    p_vec = BrownianMotion(δ, v)
                    Y_vec_f = forward(X0, p_vec, t)
                    vec_mu, vec_var = Y_vec_f.mu, Y_vec_f.var
                catch
                    ok = false
                end

                # Baseline via scalar loop
                mu_b, var_b = scalarwise_propagate(forward, X0, ()->(; δ=δ, v=v), t)

                if ok
                    @test isapprox(vec_mu, mu_b)
                    @test isapprox(vec_var, var_b)
                else
                    @test_broken true
                end

                # Backward as well
                okb = true
                vec_mu_b = nothing; vec_var_b = nothing
                try
                    p_vec = BrownianMotion(δ, v)
                    Y_vec_b = backward(X0, p_vec, t)
                    vec_mu_b, vec_var_b = Y_vec_b.mu, Y_vec_b.var
                catch
                    okb = false
                end

                mu_b2, var_b2 = scalarwise_propagate(backward, X0, ()->(; δ=δ, v=v), t)

                if okb
                    @test isapprox(vec_mu_b, mu_b2)
                    @test isapprox(vec_var_b, var_b2)
                else
                    @test_broken true
                end
            end
        end

        @testset "OrnsteinUhlenbeck param shapes" begin
            cases = [
                # (μ, v, θ)
                (0.0, 1.0, 0.5),
                (δ_last, 0.8, 0.7),
                (0.1, v_mid, 0.9),
                (0.2, 0.6, δ_last),
                (full_grid, 0.3, 0.4),
                (0.0, full_grid, 0.5)
            ]
            for (μ, v, θ) in cases
                ok = true
                vec_mu = nothing; vec_var = nothing
                try
                    p_vec = OrnsteinUhlenbeck(μ, v, θ)
                    Y_vec_f = forward(X0, p_vec, t)
                    vec_mu, vec_var = Y_vec_f.mu, Y_vec_f.var
                catch
                    ok = false
                end

                mu_b, var_b = scalarwise_propagate(forward, X0, ()->(; μ=μ, v=v, θ=θ), t)

                if ok
                    @test isapprox(vec_mu, mu_b)
                    @test isapprox(vec_var, var_b)
                else
                    @test_broken true
                end

                okb = true
                vec_mu_b = nothing; vec_var_b = nothing
                try
                    p_vec = OrnsteinUhlenbeck(μ, v, θ)
                    Y_vec_b = backward(X0, p_vec, t)
                    vec_mu_b, vec_var_b = Y_vec_b.mu, Y_vec_b.var
                catch
                    okb = false
                end

                mu_b2, var_b2 = scalarwise_propagate(backward, X0, ()->(; μ=μ, v=v, θ=θ), t)

                if okb
                    @test isapprox(vec_mu_b, mu_b2)
                    @test isapprox(vec_var_b, var_b2)
                else
                    @test_broken true
                end
            end
        end
    end

end