"""
    abstract type State end

Base type for all state representations.
"""
abstract type State end
"""
    abstract type StateLikelihood end

Base type for probability distributions over states.
"""
abstract type StateLikelihood end

abstract type DiscreteStateLikelihood <: StateLikelihood end
abstract type ContinuousStateLikelihood <: StateLikelihood end

struct DiscreteState{A} <: State
    K::Int
    state::A
end

"""
    CategoricalLikelihood(dist::AbstractArray, log_norm_const::AbstractArray)
    CategoricalLikelihood(K::Int, dims...; T=Float64)
    CategoricalLikelihood(dist::AbstractArray)

Probability distribution over discrete states.

# Parameters
- `dist`: Probability masses for each state
- `log_norm_const`: Log normalization constants
- `K`: Number of categories (for initialization)
- `dims`: Additional dimensions for initialization
- `T`: Numeric type (default: Float64)
"""
struct CategoricalLikelihood{T<:Real} <: DiscreteStateLikelihood
    dist::AbstractArray{T}
    log_norm_const::AbstractArray{T}
end

CategoricalLikelihood(K::Int, dims...; T = Float64) = CategoricalLikelihood(zeros(T, K, dims...), zeros(T, dims...))
CategoricalLikelihood(dist::AbstractArray{T}) where T<:Real = CategoricalLikelihood(dist, zeros(T, size(dist)[2:end]))

"""
    ContinuousState(state::AbstractArray{<:Real})

Representation of continuous states.

# Parameters
- `state`: Array of current state values

# Examples
```julia
# Create a continuous state
state = ContinuousState(randn(100))
```
"""
struct ContinuousState{T<:Real} <: State
    state::AbstractArray{T}
end

"""
    GaussianLikelihood(mu::AbstractArray, var::AbstractArray, log_norm_const::AbstractArray)

Gaussian probability distribution over continuous states.

# Parameters
- `mu`: Mean values
- `var`: Variances
- `log_norm_const`: Log normalization constants
"""
struct GaussianLikelihood{T<:Real} <: ContinuousStateLikelihood
    mu::AbstractArray{T}
    var::AbstractArray{T}
    log_norm_const::AbstractArray{T}
end

"""
    ⊙(a::CategoricalLikelihood, b::CategoricalLikelihood; norm=true)
    ⊙(a::GaussianLikelihood, b::GaussianLikelihood)

Compute the pointwise product of two likelihood distributions.
For Gaussian likelihoods, this results in another Gaussian.
For categorical likelihoods, this results in another categorical distribution.

# Parameters
- `a`, `b`: Input likelihood distributions
- `norm`: Whether to normalize the result (categorical only, default: true)

# Returns
A new likelihood distribution of the same type as the inputs.
"""
function ⊙(a::CategoricalLikelihood, b::CategoricalLikelihood; norm = true)
    r = CategoricalLikelihood(a.dist .* b.dist, a.log_norm_const .+ b.log_norm_const)
    if norm
        scale = sum(r.dist, dims = 1)
        r.dist ./= scale
        r.log_norm_const .+= dropdims(log.(scale), dims=1)
    end
    return r
end

function ⊙(a::GaussianLikelihood, b::GaussianLikelihood)
    res = pointwise_gaussians_product.(a.mu, a.var, b.mu, b.var)
    return GaussianLikelihood(first.(res), (x -> x[2]).(res), last.(res) .+ a.log_norm_const .+ b.log_norm_const)
end

Base.rand(d::GaussianLikelihood) = ContinuousState(rand.(Normal.(d.mu, sqrt.(d.var))))
Base.rand(d::CategoricalLikelihood) = DiscreteState(size(d.dist,1), rand.(Categorical.(sumnorm.(eachslice(d.dist, dims=Tuple(2:ndims(d.dist)))))))

"""
    stochastic(o::State)
    stochastic(T::Type, o::State)

Convert a state to its corresponding likelihood distribution:
A zero-variance (ie. delta function) Gaussian for the continuous case, and a one-hot categorical distribution for the discrete case.

# Parameters
- `o`: Input state
- `T`: Numeric type for the resulting distribution (default: Float64)

# Returns
A likelihood distribution corresponding to the input state.
"""
stochastic(T::Type, o::ContinuousState) = GaussianLikelihood(T.(o.state), T.(o.state .* 0), T.(o.state .* 0))
stochastic(o::ContinuousState{T}) where T = stochastic(T, o)

function stochastic(T::Type, o::DiscreteState)
    s = CategoricalLikelihood(zeros(T, o.K, size(o.state)...),zeros(T, size(o.state)...))
    for i in CartesianIndices(o.state)
        s.dist[o.state[i],i] = 1
    end
    return s
end
stochastic(o::State) = stochastic(Float64, o)

Base.copy(d::DiscreteState) = DiscreteState(d.K, copy(d.state))
Base.copy(d::CategoricalLikelihood) = CategoricalLikelihood(copy(d.dist), copy(d.log_norm_const))
Base.copy(d::ContinuousState) = ContinuousState(copy(d.state))
Base.copy(d::GaussianLikelihood) = GaussianLikelihood(copy(d.mu), copy(d.var), copy(d.log_norm_const))

"""
    tensor(d::Union{State, StateLikelihood})

Convert a state or likelihood to its tensor (ie. multidimensional array) representation.

# Returns
The underlying array representation of the state or likelihood.
"""
tensor(d::State) = flatview(d.state)
tensor(d::CategoricalLikelihood) = d.dist
tensor(d::GaussianLikelihood) = d.mu
tensor(d::AbstractArray) = flatview(d)
tensor(d::Real) = d