abstract type State end
abstract type StateLikelihood end
abstract type DiscreteStateLikelihood <: StateLikelihood end
abstract type ContinuousStateLikelihood <: StateLikelihood end

struct DiscreteState{T} <: State where T <: Integer
    K::Int
    state::AbstractArray{T}
end

struct CategoricalLikelihood{T} <: DiscreteStateLikelihood where T <: Real
    dist::AbstractArray{T}
    log_norm_const::AbstractArray{T}
end

CategoricalLikelihood(K::Int, dims...; T = Float64) = CategoricalLikelihood(zeros(T, K, dims...), zeros(T, dims...))

struct ContinuousState{T} <: State where T <: Real
    state::AbstractArray{T}
end

struct GaussianLikelihood{T} <: ContinuousStateLikelihood where T <: Real
    mu::AbstractArray{T}
    var::AbstractArray{T}
    log_norm_const::AbstractArray{T}
end

function ⊙(a::CategoricalLikelihood, b::CategoricalLikelihood; norm = true)
    r = CategoricalLikelihood(a.dist .* b.dist, a.log_norm_const .+ b.log_norm_const)
    if norm
        scale = dropdims(sum(r.dist, dims = 1), dims=1)
        r.dist ./= scale
        r.log_norm_const .+= log.(scale)
    end
    return r
end

function ⊙(a::GaussianLikelihood, b::GaussianLikelihood)
    res = pointwise_gaussians_product.(a.mu, a.var, b.mu, b.var)
    return GaussianLikelihood(first.(res), (x -> x[2]).(res), last.(res) .+ a.log_norm_const .+ b.log_norm_const)
end

import Base.rand
rand(d::GaussianLikelihood) = ContinuousState(rand.(Normal.(d.mu, sqrt.(d.var))))
rand(d::CategoricalLikelihood) = DiscreteState(size(d.dist,1), rand.(Categorical.(sumnorm.(eachslice(d.dist, dims=Tuple(2:ndims(d.dist)))))))

"""
    stochastic(o::State)
    stochastic(T::Type, o::State)

Converts a state to a distribution. Default type is Float64. 
"""
stochastic(T::Type, o::ContinuousState) = GaussianLikelihood(T.(o.state), T.(o.state .* 0), T.(o.state .* 0))

function stochastic(T::Type, o::DiscreteState)
    s = CategoricalLikelihood(zeros(T, o.K, size(o.state)...),zeros(T, size(o.state)...))
    for i in CartesianIndices(o.state)
        s.dist[o.state[i],i] = 1
    end
    return s
end
stochastic(o::State) = stochastic(Float64, o)

import Base.copy
copy(d::DiscreteState) = DiscreteState(d.K, copy(d.state))
copy(d::CategoricalLikelihood) = CategoricalLikelihood(copy(d.dist), copy(d.log_norm_const))
copy(d::ContinuousState) = ContinuousState(copy(d.state))
copy(d::GaussianLikelihood) = GaussianLikelihood(copy(d.mu), copy(d.var), copy(d.log_norm_const))

tensor(d::State) = flatview(d.state)
tensor(d::CategoricalLikelihood) = d.dist
tensor(d::GaussianLikelihood) = d.mu