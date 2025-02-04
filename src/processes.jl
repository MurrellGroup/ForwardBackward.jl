"""
    abstract type Process end

Base type for all stochastic processes in the package.
"""
abstract type Process end

"""
    Deterministic()

A deterministic process where endpoint conditioning results in linear interpolation between states.
"""
struct Deterministic <: Process end

"""
    abstract type DiscreteProcess <: Process end

Base type for processes with discrete state spaces.
"""
abstract type DiscreteProcess <: Process end

"""
    abstract type ContinuousProcess <: Process end

Base type for processes with continuous state spaces.
"""
abstract type ContinuousProcess <: Process end

"""
    BrownianMotion(δ::Real, v::Real)
    BrownianMotion()

Brownian motion process with drift `δ` and variance `v`.

# Parameters
- `δ`: Drift parameter (default: 0.0)
- `v`: Variance parameter (default: 1.0)

# Examples
```julia
# Standard Brownian motion
process = BrownianMotion()

# Brownian motion with drift 0.5 and variance 2.0
process = BrownianMotion(0.5, 2.0)
```
"""
struct BrownianMotion{T} <: ContinuousProcess where T <: Real
    δ::T
    v::T
end

BrownianMotion() = BrownianMotion(0.0, 1.0)

"""
    OrnsteinUhlenbeck(μ::Real, v::Real, θ::Real)
    OrnsteinUhlenbeck()

Ornstein-Uhlenbeck process with mean `μ`, variance `v`, and mean reversion rate `θ`.

# Parameters
- `μ`: Long-term mean (default: 0.0)
- `v`: Variance parameter (default: 1.0)
- `θ`: Mean reversion rate (default: 1.0)

# Examples
```julia
# Standard OU process
process = OrnsteinUhlenbeck()

# OU process with custom parameters
process = OrnsteinUhlenbeck(1.0, 0.5, 2.0)
```
"""
struct OrnsteinUhlenbeck{T} <: ContinuousProcess
    μ::T
    v::T
    θ::T
end

OrnsteinUhlenbeck() = OrnsteinUhlenbeck(0.0, 1.0, 1.0)

"""
    UniformDiscrete(μ::Real)
    UniformDiscrete()

Discrete process with uniform transition rates between states, scaled by `μ`.

# Parameters
- `μ`: Rate scaling parameter (default: 1.0)
"""
struct UniformDiscrete{T} <: DiscreteProcess
    μ::T
end

UniformDiscrete() = UniformDiscrete(1.0)

#Subs/t at equilibrium is zero, so we scale this such that, when everything is masked, μ=1 => subs/t=1
"""
    UniformUnmasking(μ::Real)
    UniformUnmasking()

Mutates only a mask (the last state index) to any other state (with equal rates). When everything is masked, 
`μ=1` corresponds to one substitution per unit time.

# Parameters
- `μ`: Rate parameter (default: 1.0)
"""
struct UniformUnmasking{T} <: DiscreteProcess
    μ::T
end

UniformUnmasking() = UniformUnmasking(1.0)

"""
    GeneralDiscrete(Q::Matrix)

Discrete process with arbitrary transition rate matrix `Q`.

# Parameters
- `Q`: Transition rate matrix
"""
struct GeneralDiscrete{T} <: DiscreteProcess
    Q::Matrix{T}
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