sumnorm(m::AbstractVector) = m ./ sum(m)

#From the first section of http://www.tina-vision.net/docs/memos/2003-003.pdf
function pointwise_gaussians_product(g1_mu::T, g1_var::T, g2_mu::T, g2_var::T) where T <: Real
    if g1_var == 0 && g2_var == 0 && g1_mu != g2_mu
        error("both gaussians have 0 variance but different means")
    elseif g1_var == 0
        return g1_mu, g1_var, logpdf(Normal(g2_mu, sqrt(g2_var)), g1_mu)
    elseif g2_var == 0
        return g2_mu, g2_var, logpdf(Normal(g1_mu, sqrt(g1_var)), g2_mu)
    end
    if g1_var == Inf && g2_var == Inf
        return (g1_mu + g2_mu) / 2, T(Inf), T(0)
    elseif g1_var == Inf
        return g2_mu, g2_var, T(0)
    elseif g2_var == Inf
        return g1_mu, g1_var, T(0)
    end
    r_var = 1 / (1 / g1_var + 1 / g2_var)
    r_mu = r_var * (g1_mu / g1_var + g2_mu / g2_var)
    r_log_norm_const =
        T(-0.5 * (
            log(2 * pi * (g1_var * g2_var / r_var)) +
            (g1_mu^2 / g1_var) +
            (g2_mu^2 / g2_var) - (r_mu^2 / r_var)
        ))
    return r_mu, r_var, r_log_norm_const
end

# -------------------------
# OU with time-varying variance v(t)
# -------------------------
function _ou_noise_Q(t1, t2, θ, a0, w::AbstractVector, β::AbstractVector)
    Δ = t2 .- t1
    if iszero(θ)
        Q = a0 .* Δ
        for k in eachindex(w, β)
            bk = β[k]
            wk = w[k]
            if iszero(bk)
                Q = Q .+ wk .* Δ
            else
                Q = Q .+ wk .* (exp.(bk .* t2) .- exp.(bk .* t1)) ./ bk
            end
        end
        return Q
    else
        e_m2θΔ = exp.(-2 .* θ .* Δ)
        Q = a0 .* (-expm1.(-2 .* θ .* Δ)) ./ (2 .* θ)
        for k in eachindex(w, β)
            bk = β[k]
            wk = w[k]
            denom = bk .+ 2 .* θ
            # denom is scalar if θ is scalar; handle near-zero safely
            if abs(float(denom)) > eps(float(denom))
                Q = Q .+ (wk ./ denom) .* (exp.(bk .* t2) .- exp.(bk .* t1) .* e_m2θΔ)
            else
                Q = Q .+ wk .* exp.(bk .* t2) .* Δ
            end
        end
        return Q
    end
end