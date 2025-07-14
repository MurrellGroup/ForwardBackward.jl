#using Flux: onecold

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
    BrownianMotion(δ::T, v::T) where T <: Real
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
struct BrownianMotion{T} <: ContinuousProcess where T <: Real
    δ::T
    v::T
end

BrownianMotion() = BrownianMotion(0, 1)
BrownianMotion(v::T) where T = BrownianMotion(T(0), v)


"""
    OrnsteinUhlenbeck(μ::T, v::T, θ::T) where T <: Real
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
struct OrnsteinUhlenbeck{T} <: ContinuousProcess
    μ::T
    v::T
    θ::T
end

OrnsteinUhlenbeck() = OrnsteinUhlenbeck(0, 1, 1)

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

abstract type Nodal end

mutable struct PiNode <: Nodal
    u::Float64
    parent::Union{PiNode,Nothing}
    children::Union{Vector{<:Nodal},Nothing}
    leaf_indices::Union{Vector{<:Int}, Nothing}
    PiNode(u) = new(u, nothing, nothing, nothing)
end

mutable struct PiLeaf <: Nodal
    index::Int64
    parent::Union{PiNode, Nothing}
    PiLeaf(index) = new(index, nothing)
end

struct HPiQ{T} <: DiscreteProcess
    tree::PiNode
    π::Vector{T}
end

function add_child!(node::PiNode, child::Nodal)
    if isnothing(node.children)
        node.children = typeof(child)[]
    end
    push!(node.children, child)
    child.parent = node
end

function init_leaf_indices!(node::PiNode)
    indices = Int[]
    if isnothing(node.children)
        node.leaf_indices = indices
        return indices
    end
    for child in node.children
        if isa(child, PiLeaf)
            push!(indices, child.index)
        elseif isa(child, PiNode)
            append!(indices, init_leaf_indices!(child))
        end
    end
    node.leaf_indices = sort!(unique!(indices))
    return node.leaf_indices
end

# --- Q Matrix Generation Function (from get_q_function artifact) ---

function get_all_nodes!(node::PiNode, nodes::Vector{PiNode})
    push!(nodes, node)
    if !isnothing(node.children)
        for child in node.children
            if isa(child, PiNode)
                get_all_nodes!(child, nodes)
            end
        end
    end
    return nodes
end

function get_Q(process::HPiQ)
    (; tree, π) = process
    N = length(π)
    Q = zeros(Float64, N, N)
    all_nodes = PiNode[]
    get_all_nodes!(tree, all_nodes)

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        length(idx) <= 1 && continue
        u = node.u
        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue
        for i_global in idx
            for j_global in idx
                if i_global != j_global
                    Q[i_global, j_global] += u * (π[j_global] / sum_π)
                end
            end
        end
    end

    for i in 1:N
        Q[i, i] = -sum(Q[i, :])
    end
    return Q
end

# function get_Q_row(process::HPiQ, Xt::AbstractArray{T}) where T
#     (; tree, π) = process
#     N = length(π)
#     Q = zeros(Float64, size(Xt)...)
#     all_nodes = PiNode[]
#     get_all_nodes!(tree, all_nodes)
#     batch_indices = onecold(Xt) 
#     for node in all_nodes
#         isnothing(node.leaf_indices) && continue
#         idx = node.leaf_indices
#         length(idx) <= 1 && continue
#         u = node.u
#         π_partition_view = view(π, idx)
#         sum_π = sum(π_partition_view)
#         isapprox(sum_π, 0.0) && continue

#         for I in CartesianIndices(batch_indices)
#             for j_global in idx
#                 if batch_indices[I] != j_global && batch_indices[I] in idx
#                     Q[j_global, I[1], I[2]] += u * (π[j_global] / sum_π)
#                 end
#             end
#         end
#     end
#     return Q
# end
