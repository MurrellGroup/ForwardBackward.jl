"""
    abstract type Process

Base type for all processes defined in the ForwardBackward package.
"""
abstract type Process end

"""
    Deterministic()

A deterministic process where endpoint conditioning results in linear interpolation between states.
"""
struct Deterministic <: Process end

"""
    abstract type DiscreteProcess <: Process

Base type for processes with discrete state spaces.
"""
abstract type DiscreteProcess <: Process end

"""
    abstract type ContinuousProcess <: Process

Base type for processes with continuous state spaces.
"""
abstract type ContinuousProcess <: Process end

"""
    BrownianMotion(δ::T1, v::T2) where T1 <: Real where T2 <: Real
    BrownianMotion(v::Real)
    BrownianMotion()

Brownian motion process with drift `δ` and variance `v`.

# Parameters
- `δ`: Drift parameter (default: 0)
- `v`: Variance parameter (default: 1)

# Tip
The rate parameters must match the type of the state (eg. Float32 both both process and state), but you can avoid this if you use integer process parameters.

# Examples
```julia
# Standard Brownian motion
process = BrownianMotion()

# Brownian motion with drift 0.5 and variance 2.0
process = BrownianMotion(0.5, 2.0)
```
"""
struct BrownianMotion{T1,T2} <: ContinuousProcess where T1 <: Real where T2 <: Real
    δ::T1
    v::T2
end

BrownianMotion() = BrownianMotion(0, 1)
BrownianMotion(v::T) where T = BrownianMotion(T(0), v)


"""
    OrnsteinUhlenbeck(μ::T1, v::T2, θ::T3) where T1 <: Real where T2 <: Real where T3 <: Real
    OrnsteinUhlenbeck()

Ornstein-Uhlenbeck process with mean `μ`, variance `v`, and mean reversion rate `θ`.

# Parameters
- `μ`: Long-term mean (default: 0)
- `v`: Variance parameter (default: 1)
- `θ`: Mean reversion rate (default: 1)

# Tip
The rate parameters must match the type of the state (eg. Float32 both both process and state), but you can avoid this if you use integer process parameters.

# Examples
```julia
# Standard OU process
process = OrnsteinUhlenbeck()

# OU process with custom parameters
process = OrnsteinUhlenbeck(1.0, 0.5, 2.0)
```
"""
struct OrnsteinUhlenbeck{T1,T2,T3} <: ContinuousProcess
    μ::T1
    v::T2
    θ::T3
end

OrnsteinUhlenbeck() = OrnsteinUhlenbeck(0, 1, 1)



"""
    OrnsteinUhlenbeckExpVar(μ, θ::Real, a0::Real, w::AbstractVector{<:Real}, β::AbstractVector{<:Real})
    OrnsteinUhlenbeckExpVar()
    OrnsteinUhlenbeckExpVar(μ, θ, v)
    OrnsteinUhlenbeckExpVar(μ, θ, v_at_0, v_at_1; dec = -0.1)

Ornstein–Uhlenbeck process with time-varying instantaneous variance
v(t) = a0 + sum(w[k] * exp(β[k] * t) for k).

Parameters
- μ  : Long-run mean (default 0)
- θ  : Mean-reversion rate (default 1)
- a0 : Baseline variance level (default 1)
- w  : Weights of exponential components (default empty)
- β  : Exponents (same length as w) (default empty)

Notes
- Keep w ≥ 0 and a0 ≥ 0 if you want v(t) ≥ 0.
- Handles the limits θ → 0 and β[k] + 2θ → 0 in a numerically stable way.
- OrnsteinUhlenbeckExpVar(μ, θ, v_at_0, v_at_1; dec = -0.1) sets up a process where the variance decays nearly linearly from v_at_0 to v_at_1 over the interval [0, 1].
"""
struct OrnsteinUhlenbeckExpVar{M, Tθ<:Real, Ta0<:Real, Vw<:AbstractVector{<:Real}, Vβ<:AbstractVector{<:Real}} <: ContinuousProcess
    μ::M
    θ::Tθ
    a0::Ta0
    w::Vw
    β::Vβ
    function OrnsteinUhlenbeckExpVar(μ, θ::Tθ, a0::Ta0, w::Vw, β::Vβ) where {Tθ<:Real, Ta0<:Real, Vw<:AbstractVector{<:Real}, Vβ<:AbstractVector{<:Real}}
        length(w) == length(β) || throw(ArgumentError("w and β must have the same length"))
        new{typeof(μ), Tθ, Ta0, Vw, Vβ}(μ, θ, a0, w, β)
    end
end

OrnsteinUhlenbeckExpVar() = OrnsteinUhlenbeckExpVar(0.0, 1.0, 1.0, Float64[], Float64[])
OrnsteinUhlenbeckExpVar(μ, θ, v) = OrnsteinUhlenbeckExpVar(μ, θ, v, eltype(μ)[], eltype(μ)[])
function OrnsteinUhlenbeckExpVar(μ, θ, v_at_0, v_at_1; dec = -0.1)
    @assert v_at_0 > 0
    @assert v_at_1 > 0
    @assert dec < 0
    OrnsteinUhlenbeckExpVar(μ, θ, v_at_1-((((v_at_0-v_at_1)*exp(dec))/(1-exp(dec)))), [(((v_at_0-v_at_1))/(1-exp(dec)))], [dec])
end

"""
    UniformDiscrete(μ::Real)
    UniformDiscrete()

Discrete process with uniform transition rates between states, scaled by `μ`.

# Parameters
- `μ`: Rate scaling parameter (default: 1)

# Tip
The rate parameters must match the type of the state (eg. Float32 both both process and state), but you can avoid this if you use integer process parameters.
"""
struct UniformDiscrete{T} <: DiscreteProcess
    μ::T
end

UniformDiscrete() = UniformDiscrete(1)

#Subs/t at equilibrium is zero, so we scale this such that, when everything is masked, μ=1 => subs/t=1
"""
    UniformUnmasking(μ::Real)
    UniformUnmasking()

Mutates only a mask (the last state index) to any other state (with equal rates). When everything is masked, 
`μ=1` corresponds to one substitution per unit time.

# Parameters
- `μ`: Rate parameter (default: 1)

# Tip
The rate parameters must match the type of the state (eg. Float32 both both process and state), but you can avoid this if you use integer process parameters.
"""
struct UniformUnmasking{T} <: DiscreteProcess
    μ::T
end

UniformUnmasking() = UniformUnmasking(1)

"""
    GeneralDiscrete(Q::Matrix)

Discrete process with arbitrary transition rate matrix `Q`.

# Parameters
- `Q`: Transition rate matrix
"""
struct GeneralDiscrete{A<:AbstractMatrix{<:Real}} <: DiscreteProcess
    Q::A
end

"""
    PiQ(r::Real, π::Vector{<:Real}; normalize=true)
    PiQ(π::Vector{<:Real}; normalize=true)

Discrete process that switches to states proportionally to the stationary distribution `π` with rate `r`.

# Parameters
- `r`: Overall switching rate (default: 1.0)
- `π`: Target stationary distribution (will always be normalized to sum to 1)
- `normalize`: Whether to normalize the expected substitutions per unit time to be 1 when `r` = 1 (default: true)

# Examples
```julia
# Process with uniform stationary distribution
process = PiQ(ones(4) ./ 4)

# Process with custom stationary distribution and rate
process = PiQ(2.0, [0.1, 0.2, 0.3, 0.4])
```
"""
struct PiQ{T} <: DiscreteProcess
    r::T
    π::Vector{T}
    β::T
end

function PiQ(r::T,π::Vector{T}; normalize=true) where T <: Real
    piNormed = π ./ sum(π)
    β = normalize ? 1/(1-sum(abs2.(piNormed))) : T(1.0)
    PiQ(r, piNormed, β)
end

PiQ(π::Vector{T}; normalize=true) where T <: Real = PiQ(T(1.0), π; normalize=normalize)