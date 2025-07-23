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

"""
    abstract type Nodal <: Nodal

    
    Base type for tree nodes which is used to define the PHiQ process.
"""
abstract type Nodal end

"""
    mutable struct PiNode{T} <: Nodal

Internal node type for The PHiQ tree.
    
# Parameters
- `u`: Rate parameter
- `parent`: Parent node
- `children`: Children nodes
- `leaf_indices`: State indices of descendent leaf nodes
"""
mutable struct PiNode{T} <: Nodal
    u::T
    parent::Union{PiNode{T}, Nothing}
    children::Union{Vector{<:Nodal},Nothing}
    leaf_indices::Union{Vector{<:Int}, Nothing}
    PiNode(u::T) where T = new{T}(u, nothing, nothing, nothing)
end

"""
    mutable struct PiLeaf{T} <: Nodal

A PiLeaf node is a representation of a discrete state.

# Parameters
- `index`: State index
- `parent`: parent node
"""
mutable struct PiLeaf <: Nodal
    index::Int64
    parent::Union{PiNode, Nothing}
    PiLeaf(index) = new(index, nothing)
end

"""
    struct HPiQ{T} <: DiscreteProcess
    
Discrete-state continuous-time process with an equilibrium vector `π` and a hierichal tree structure `tree`, which imposes a hierichal structure where transition events can occur for a subset of the states. 
Note, remember to call `init_leaf_indicies!` to correctly collect descendent leaf states for internal nodes, this is needed to call e.g. forward and backward.

# Parameters
- `tree`: Root node of a tree
- `π`: equilibrium vector

# Examples
```julia

# The root
tree = PiNode(1.0) 

#Internal Nodes
child1 = PiNode(2.0) 
child2 = PiNode(3.0)

add_child!(tree, child1)
add_child!(tree, child2)

# States
leaf1 = PiLeaf(1)
leaf2 = PiLeaf(2)
leaf3 = PiLeaf(3)
leaf4 = PiLeaf(4)

add_child!(child1, leaf1)
add_child!(child1, leaf2)
add_child!(child2, leaf3)
add_child!(child2, leaf4)

init_leaf_indices!(tree)
π = [0.2, 0.3, 0.4, 0.1]

HPiQ_process = HPiQ(tree, π)
```
"""
struct HPiQ{T} <: DiscreteProcess
    tree::PiNode
    π::Vector{T}
end

"""
    add_child!(node::PiNode, child::Nodal)

Helper function for the construction of a HPiQ tree.
    
# Parameters
- `node`: The parent node
- `child`: The child node 
"""
function add_child!(node::PiNode, child::Nodal)
    if isnothing(node.children)
        node.children = Nodal[]
    end
    push!(node.children, child)
    child.parent = node
end

"""
    init_leaf_indices!(node::PiNode)

This function assigns the state indices of its descendent leaf nodes to each internal node in a HPiQ tree.

# Parameters
- `node`: The root node of the tree 
"""
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

# Gets all internal nodes of a HPiQ tree
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

# This maps HPiQ process to its corresponding transition rate matrix.
function HPiQ_Qmatrix(process::HPiQ)
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
