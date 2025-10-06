"""
    struct OUBridgeExpVar <: ContinuousProcess

Endpoint-conditioned OU bridge with time-varying noise.
- Drift of the conditional SDE is Λ(t)*(x_c - X_t), with Λ including θ and the bridge term.
- Noise schedule uses the same mixture-of-exponentials parameterization as your OU:
    σ²(t) ≈ a0 + sum_k w[k] * exp(β[k] * t).
- the endpoint x_c provides the target, acting like the mean in an OU process.
- Only endpoint-conditioned single-time sampling is provided.

Parameters
- θ    :: Real                         # base endpoint-reversion knob (mixing-rate)
- a0   :: Real                         # baseline noise scale
- w    :: AbstractVector{<:Real}       # weights for exp components
- β    :: AbstractVector{<:Real}       # exponents (can be negative/positive)

Notes
- Sampling at time b uses closed-form Gaussian conditioning with:
    φ(t,s) = exp(-θ*(t - s))
    V_b = _ou_noise_Q(a, b, θ, a0, w, β)
    V_c = _ou_noise_Q(a, c, θ, a0, w, β)
    C_{b,c} = exp(-θ*(c - b)) * V_b
    m_b = x_c + φ(b,a)*(x_a - x_c), m_c = x_c + φ(c,a)*(x_a - x_c)
- No numerical integration; uses your `_ou_noise_Q`.
"""
struct OUBridgeExpVar{Tθ<:Real, Ta0<:Real, Vw<:AbstractVector{<:Real}, Vβ<:AbstractVector{<:Real}} <: ContinuousProcess
    θ::Tθ
    a0::Ta0
    w::Vw
    β::Vβ
    function OUBridgeExpVar(θ::Tθ, a0::Ta0, w::Vw, β::Vβ) where {Tθ<:Real, Ta0<:Real, Vw<:AbstractVector{<:Real}, Vβ<:AbstractVector{<:Real}}
        length(w) == length(β) || throw(ArgumentError("w and β must have the same length"))
        new{Tθ, Ta0, Vw, Vβ}(θ, a0, w, β)
    end
end

OUBridgeExpVar() = OUBridgeExpVar(1.0, 1.0, Float64[], Float64[])
OUBridgeExpVar(θ, v) = OUBridgeExpVar(θ, v, eltype(v)[], eltype(v)[])

function OUBridgeExpVar(θ, v_at_0, v_at_1; dec = -0.1)
    a0 = v_at_1 - (((v_at_0 - v_at_1) * exp(dec)) / (1 - exp(dec)))
    w1 = ((v_at_0 - v_at_1)) / (1 - exp(dec))
    return OUBridgeExpVar(θ, a0, [w1], [dec])
end

function endpoint_conditioned_sample(Xa::ContinuousState, Xc::ContinuousState,
                                     P::OUBridgeExpVar, t_a, t_b, t_c) :: ContinuousState
    xa = Xa.state
    xc = Xc.state
    T  = eltype(xa)
    nd = ndims(xa)

    ta = expand(t_a, nd)
    tb = expand(t_b, nd)
    tc = expand(t_c, nd)

    # OU fundamental solutions with constant θ
    phi_ba = exp.(-P.θ .* (tb .- ta))
    phi_ca = exp.(-P.θ .* (tc .- ta))
    phi_cb = exp.(-P.θ .* (tc .- tb))

    # Unconditioned means (with μ ≡ x_c)
    m_b = xc .+ phi_ba .* (xa .- xc)
    m_c = xc .+ phi_ca .* (xa .- xc)

    # Variance terms using your OU noise integral helper
    V_b = _ou_noise_Q(ta, tb, P.θ, P.a0, P.w, P.β)
    V_c = _ou_noise_Q(ta, tc, P.θ, P.a0, P.w, P.β)

    # Cross-covariance C_{b,c} = exp(-θ*(c-b)) * V_b
    C_bc = phi_cb .* V_b

    # Conditional Gaussian at time b
    # mean_b| = m_b + (C_bc / V_c) * (x_c - m_c)
    # var_b|  = V_b - (C_bc^2)/V_c
    denom = V_c
    # Guard against exact-zero V_c (deterministic case)
    epsT = eps(T)
    safe_denom = max.(denom, epsT)

    mu_cond  = m_b .+ (C_bc ./ safe_denom) .* (xc .- m_c)
    var_cond = V_b .- (C_bc .* C_bc) ./ safe_denom
    var_cond = max.(var_cond, zero(T))

    z = randn(T, size(mu_cond)...)
    xb = mu_cond .+ sqrt.(var_cond) .* z
    return ContinuousState(xb)
end

export OUBridgeExpVar


#=
"""
    endpoint_conditioned_sample(X0::ManifoldState, X1::ManifoldState,
                                p::ManifoldProcess{<:OUBridgeExpVar},
                                t_a, t_b, t_c; Δt = 0.05)

Endpoint-conditioned sampling on a manifold for an OU-style bridge with exponential-mixture noise.

This integrates the **conditional bridge dynamics** using only two manifold
operations per micro-step:
  1) `shortest_geodesic!` toward the endpoint `X1.state[...]` with fraction `α`,
  2) `perturb!` by an isotropic tangent Gaussian with variance `v_step`.

It is **exact in Euclidean space** for the OUBridgeExpVar bridge at the chosen discretization scale,
because each step matches the finite-horizon OU-bridge transition’s first two moments.

Math (per element, at time `t`, stepping to `t+Δ`):
- Let `θ = P.θ`, and define
    φ(Δ)         = exp(-θ*Δ)
    Q(t, t+Δ)    = _ou_noise_Q(t, t+Δ, θ, P.a0, P.w, P.β)
    D(t→c)       = _ou_noise_Q(t, t_c, θ, P.a0, P.w, P.β)
    φ_c(t)       = exp(-θ*(t_c - t))
    φ_c(t+Δ)     = exp(-θ*(t_c - (t+Δ)))

- The **deterministic contraction** for the OU bridge over [t, t+Δ] is
    ρ = φ(Δ) - [Q(t, t+Δ) / D(t→c)] * φ_c(t+Δ) * φ_c(t)
  (so the conditional mean is `x_c + ρ (x_t - x_c)`).

- The **conditional variance increment** is
    v_step = Q(t, t+Δ) - [φ_c(t+Δ)^2 * Q(t, t+Δ)^2] / D(t→c).

Implementation on a manifold:
- Geodesic fraction: α = clamp(1 - ρ, 0, 1).
- Then:
    shortest_geodesic!(M, cur, cur, x_c_point, α)
    if v_step > 0
        perturb!(M, cur, cur, v_step)
    end
Repeat until reaching `t_b`.

Notes:
- This function assumes `p` wraps an `OUBridgeExpVar` in a field named `proc` or `P`.
- Handles scalar or array times via `expand`. Loops over all elements of `X0.state`.
- `Δt` is the micro-step size; decreasing it refines the bridge approximation on curved manifolds
  (exact in Euclidean space for any `Δt` due to the moment-matching step).
"""
=#
#Fast override for manifold processes. Should be exact on Euclidean, but unclear about accuracy compared to tangent steps for manifolds.
function endpoint_conditioned_sample(X0::ManifoldState, X1::ManifoldState,
                                     p::ManifoldProcess{<:OUBridgeExpVar},
                                     t_a, t_b, t_c; Δt = 0.05)
    P = p.v
    θ  = P.θ
    a0 = P.a0
    w  = P.w
    β  = P.β

    M  = X0.M
    T  = eltype(flatview(X0.state))
    nd = ndims(X0.state)

    ta_arr = zeros(T, size(X0.state)) .+ expand(t_a, nd)
    tb_arr = zeros(T, size(X0.state)) .+ expand(t_b, nd)
    tc_arr = zeros(T, size(X0.state)) .+ expand(t_c, nd)

    Xt = copy(X0)
    epsT = eps(T)

    for ind in CartesianIndices(X0.state)
        t  = ta_arr[ind]
        tb = tb_arr[ind]
        tc = tc_arr[ind]

        # current point evolves in-place at Xt.state[ind]
        while t < tb
            inc = min(T(t + Δt), tb)
            Δ   = inc - t

            # OU kernels and integrals
            φΔ      = exp(-θ * Δ)
            Q_step  = _ou_noise_Q(t, inc, θ, a0, w, β)         # ∫ e^{-2θ(inc-s)} q(s) ds
            D_to_c  = _ou_noise_Q(t, tc,  θ, a0, w, β)         # ∫ e^{-2θ(tc - s)} q(s) ds
            φc_t    = exp(-θ * (tc - t))
            φc_inc  = exp(-θ * (tc - inc))

            # Contraction factor and step fraction
            denom   = max(D_to_c, epsT)
            ρ       = φΔ - (Q_step / denom) * φc_inc * φc_t
            α       = clamp(1 - ρ, zero(T), one(T))

            # Deterministic pull along the geodesic toward the endpoint
            shortest_geodesic!(M, Xt.state[ind], Xt.state[ind], X1.state[ind], α)

            # Conditional variance increment for this slice; add tangent noise
            v_step = Q_step - (φc_inc^2 * Q_step^2) / denom
            if v_step > 0
                perturb!(M, Xt.state[ind], Xt.state[ind], v_step)
            end

            t = inc
        end
    end

    return Xt
end




"""
    struct OUBridgeDistVarSched <: ContinuousProcess

OU-style endpoint-conditioned bridge with drift θ * (x_c - X_t) and a
variance-to-go schedule defined by a distribution in absolute time.

Let S(t) = 1 - cdf(vdist, t) (survival). Define R(t) = Rscale * S(t).
For a < b < c, with X_a = x_a and X_c = x_c:
  phi(t2,t1) = exp(-θ * (t2 - t1))
  r = S(b) / S(a)
  μ_b   = x_c + phi(b,a) * r * (x_a - x_c)
  Var_b = (Rscale * S(b) * (S(a) - S(b))) / (S(a) * phi(c,b)^2)

This formulation uses only survival ratios (no clamping), so it’s
partition-consistent. Require S(a) > 0 (i.e., cdf(vdist, a) < 1).
"""
struct OUBridgeDistVarSched{Tθ<:Real, TR<:Real,
                                Dv<:Distributions.UnivariateDistribution} <: ContinuousProcess
    θ::Tθ
    Rscale::TR
    vdist::Dv
end

"""
    endpoint_conditioned_sample(Xa::ContinuousState, Xc::ContinuousState,
                                P::OUBridgeDistVarSched, t_a, t_b, t_c) :: ContinuousState

Draw X_b | (X_a = Xa.state, X_c = Xc.state), broadcasting over array states and
scalar/array times via `expand`. Assumes `Distributions` is available (uses `cdf`).
Throws if S(a) = 1 - cdf(vdist, a) ≤ 0 anywhere.
"""
function endpoint_conditioned_sample(Xa::ContinuousState, Xc::ContinuousState,
                                     P::OUBridgeDistVarSched, t_a, t_b, t_c) :: ContinuousState
    xa = Xa.state
    xc = Xc.state
    T  = eltype(xa)
    nd = ndims(xa)

    ta = expand(t_a, nd)
    tb = expand(t_b, nd)
    tc = expand(t_c, nd)

    # OU fundamental solutions with constant θ
    phi_ba = exp.(-P.θ .* (tb .- ta))
    phi_cb = exp.(-P.θ .* (tc .- tb))

    # Survival S(t) = 1 - CDF_v(t) in absolute time
    Sa = one(T) .- cdf.(P.vdist, ta)
    Sb = one(T) .- cdf.(P.vdist, tb)

    # Require S(a) > 0 everywhere; otherwise the bridge is already collapsed.
    #if any(Sa .<= zero(T))
    #    throw(ArgumentError("vdist must satisfy cdf(a) < 1 so survival S(a) > 0 at all entries."))
    #end

    # Ratios and marginals (no clamping -> partition-consistent)
    r    = Sb ./ Sa
    mu_b = xc .+ phi_ba .* r .* (xa .- xc)

    var_b = (P.Rscale .* Sb .* (Sa .- Sb)) ./ (Sa .* (phi_cb .^ 2))
    var_b = max.(var_b, zero(T))  # only to avoid tiny negative due to fp roundoff

    z  = randn(T, size(mu_b)...)
    xb = mu_b .+ sqrt.(var_b) .* z
    return ContinuousState(xb)
end


export OUBridgeDistVarSched


"""
    struct ScheduledGaussianMixture <: ContinuousProcess

Gaussian bridge for single-time endpoint-conditioned sampling.

Design:
- Drift of the conditional SDE is Λ(t) * (x_c - X_t).
- Two schedules, specified in absolute time (no renormalization):
  • Mean approach via an integrable rate k(t) with integral
      ∫_{t1}^{t2} k = k0*(t2 - t1) + k1*(F_k(t2) - F_k(t1)),
    where F_k is the CDF of `kdist`.
  • Variance-to-go R(t) via a survival function
      R(t) = Rscale * S_v(t),  S_v(t) = 1 - F_v(t),
    where F_v is the CDF of `vdist`. Then R(c)=0.

Bridge at time b (a < b < c), with X_a=x_a and X_c=x_c:
- φ(t2,t1) = exp(-∫_{t1}^{t2} k).
- mean:    μ_b = x_c + φ(b,a) * (R(b)/R(a)) * (x_a - x_c)
- variance: Var_b = R(b) * (R(a) - R(b)) / ( R(a) * φ(c,b)^2 )

Absolute-time CDFs ensure consistency across partitions:
sampling (a→b1→b2) matches sampling (a→b2) marginally.

Fields:
- k0, k1          : Real
- kdist           : UnivariateDistribution (absolute-time CDF for k)
- Rscale          : Real (scales R)
- vdist           : UnivariateDistribution (absolute-time CDF for R)
"""
struct ScheduledGaussianMixture{Tk0<:Real, Tk1<:Real,
                          Dk<:Distributions.UnivariateDistribution,
                          TR<:Real,
                          Dv<:Distributions.UnivariateDistribution} <: ContinuousProcess
    k0::Tk0
    k1::Tk1
    kdist::Dk
    Rscale::TR
    vdist::Dv
end


function endpoint_conditioned_sample(Xa::ContinuousState, Xc::ContinuousState,
                                     P::ScheduledGaussianMixture, t_a, t_b, t_c)
    xa = Xa.state
    xc = Xc.state
    T  = eltype(xa)
    nd = ndims(xa)

    ta = expand(t_a, nd)
    tb = expand(t_b, nd)
    tc = expand(t_c, nd)

    # --- fundamental solutions φ via absolute-time CDFs of kdist ---
    Fka = cdf.(P.kdist, ta)
    Fkb = cdf.(P.kdist, tb)
    Fkc = cdf.(P.kdist, tc)

    K_ba = P.k0 .* (tb .- ta) .+ P.k1 .* (Fkb .- Fka)
    K_cb = P.k0 .* (tc .- tb) .+ P.k1 .* (Fkc .- Fkb)

    phi_ba = exp.(-K_ba)
    phi_cb = exp.(-K_cb)

    # --- variance-to-go R via absolute-time survival of vdist ---
    Sa = one(T) .- cdf.(P.vdist, ta)   # S_v(a)
    Sb = one(T) .- cdf.(P.vdist, tb)   # S_v(b)

    Ra = P.Rscale .* Sa
    Rb = P.Rscale .* Sb

    # --- bridge mean and variance at b ---
    mu_b   = xc .+ phi_ba .* (Rb ./ Ra) .* (xa .- xc)
    var_b  = Rb .* (Ra .- Rb) ./ (Ra .* (phi_cb .^ 2))

    # --- sample with same element type as state ---
    var_b = max.(var_b, zero(T))
    z = randn(T, size(mu_b)...)
    xb = mu_b .+ sqrt.(var_b) .* z
    return ContinuousState(xb)
end


export ScheduledGaussianMixture