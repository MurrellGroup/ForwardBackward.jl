abstract type Process end

"""
    Deterministic()

Endpoint conditioning under a deterministic process is a linear interpolation.
"""
struct Deterministic <: Process end

abstract type DiscreteProcess <: Process end
abstract type ContinuousProcess <: Process end

struct BrownianMotion{T} <: ContinuousProcess where T <: Real
    δ::T
    v::T
end

BrownianMotion() = BrownianMotion(0.0, 1.0)

struct OrnsteinUhlenbeck{T} <: ContinuousProcess
    μ::T
    v::T
    θ::T
end

OrnsteinUhlenbeck() = OrnsteinUhlenbeck(0.0, 1.0, 1.0)

struct UniformDiscrete{T} <: DiscreteProcess
    μ::T
end

UniformDiscrete() = UniformDiscrete(1.0)

struct GeneralDiscrete{T} <: DiscreteProcess
    Q::Matrix{T}
end

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