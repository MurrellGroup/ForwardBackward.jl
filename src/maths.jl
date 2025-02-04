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
        -0.5 * (
            log(2 * pi * (g1_var * g2_var / r_var)) +
            (g1_mu^2 / g1_var) +
            (g2_mu^2 / g2_var) - (r_mu^2 / r_var)
        )
    return r_mu, r_var, r_log_norm_const
end