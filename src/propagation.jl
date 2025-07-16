expand(t::Real, x) = t
function expand(t::AbstractArray, d::Int)
    ndt = ndims(t)
    d - ndt < 0 && error("Cannot expand array of size $(size(t)) to $d dimensions.")
    reshape(t, ntuple(Returns(1), d - ndt)..., size(t)...)
end

"""
    forward!(Xdest::StateLikelihood, Xt::State, process::Process, t)
    forward(Xt::StateLikelihood, process::Process, t)
    forward(Xt::State, process::Process, t)

Propagate a state or likelihood forward in time according to the process dynamics.

# Parameters
- `Xdest`: Destination for in-place operation
- `Xt`: Initial state or likelihood
- `process`: The stochastic process
- `t`: Time to propagate forward

# Returns
The forward-propagated state or likelihood
"""
forward!(Xdest::StateLikelihood, Xt::State, process::Process, t) = forward!(Xdest, stochastic(eltype(t), Xt), process, t)
forward(Xt::StateLikelihood, process::Process, t) = forward!(copy(Xt), Xt, process, t)
forward(Xt::State, process::Process, t) = forward!(stochastic(eltype(t), Xt), Xt, process, t)

"""
    backward!(Xdest::StateLikelihood, Xt::State, process::Process, t)
    backward(Xt::StateLikelihood, process::Process, t)
    backward(Xt::State, process::Process, t)

Propagate a state or likelihood backward in time according to the process dynamics.

# Parameters
- `Xdest`: Destination for in-place operation
- `Xt`: Final state or likelihood
- `process`: The stochastic process
- `t`: Time to propagate backward

# Returns
The backward-propagated state or likelihood
"""
backward!(Xdest::StateLikelihood, Xt::State, process::Process, t) = backward!(Xdest, stochastic(eltype(t), Xt), process, t)
backward(Xt::StateLikelihood, process::Process, t) = backward!(copy(Xt), Xt, process, t)
backward(Xt::State, process::Process, t) = backward!(stochastic(eltype(t), Xt), Xt, process, t)

"""
    interpolate(X0::ContinuousState, X1::ContinuousState, tF, tB)

Linearly interpolate between two continuous states.

# Parameters
- `X0`: Initial state
- `X1`: Final state
- `tF`: Forward time
- `tB`: Backward time

# Returns
The interpolated state
"""
function interpolate(X0::ContinuousState, X1::ContinuousState, tF, tB)
    t0 = @. tF/(tF + tB)
    t1 = @. 1 - t0
    return ContinuousState(X0.state .* expand(t1, ndims(X0.state)) .+ X1.state .* expand(t0, ndims(X1.state)))
end

"""
    endpoint_conditioned_sample(X0, X1, p, tF, tB)
    endpoint_conditioned_sample(X0, X1, p, t)
    endpoint_conditioned_sample(X0, X1, p::Deterministic, tF, tB)

Generate a sample from the endpoint-conditioned process.

# Parameters
- `X0`: Initial state
- `X1`: Final state
- `p`: The stochastic process
- `t`, `tF`: Forward time
- `tB`: Backward time (defaults to 1-t for single time parameter)

# Returns
A sample from the endpoint-conditioned distribution

# Notes
For continuous processes, uses the forward-backward algorithm.
For deterministic processes, uses linear interpolation.
"""
endpoint_conditioned_sample(X0, X1, p, tF, tB) = rand(forward(X0, p, tF) ⊙ backward(X1, p, tB))
endpoint_conditioned_sample(X0, X1, p, t) = endpoint_conditioned_sample(X0, X1, p, t, clamp.(1 .- t, 0, 1))
endpoint_conditioned_sample(X0, X1, p::Deterministic, tF, tB) = interpolate(X0, X1, tF, tB)

function endpoint_conditioned_dist(X0, X1, p, tF, tB)
    f = forward(X0, p, tF)
    b = backward(X1, p, tB)
    return f ⊙ b
end

function forward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::OrnsteinUhlenbeck, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    μ, v, θ = process.μ, process.v, process.θ
    @. x_dest.mu = μ + exp(-θ * t) * (Xt.mu - μ)
    @. x_dest.var = exp(-2θ * t) * Xt.var + (v / (2θ)) * (1 - exp(-2θ * t))
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function backward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::OrnsteinUhlenbeck, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    μ, v, θ = process.μ, process.v, process.θ
    @. x_dest.mu = μ + exp(θ * t) * (Xt.mu - μ)
    @. x_dest.var = exp(2θ * t) * (Xt.var + (v / (2θ)) * (1 - exp(-2θ * t)))
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function forward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::BrownianMotion, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    x_dest.mu .= @. Xt.mu + process.δ * t
    x_dest.var .= @. process.v * t + Xt.var
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function backward!(x_dest::GaussianLikelihood, Xt::GaussianLikelihood, process::BrownianMotion, elapsed_time)
    t = expand(elapsed_time, ndims(Xt.mu))
    x_dest.mu .= @. Xt.mu - process.δ * t
    x_dest.var .= @. process.v * t + Xt.var
    x_dest.log_norm_const .= Xt.log_norm_const
    return x_dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::PiQ, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    scals = sum(source.dist, dims = 1)
    pow = @. exp(-process.β * process.r * t)
    c1 = @. (1 - pow) * process.π
    c2 = @. pow + (1 - pow) * process.π
    dest.dist .= @. (scals - source.dist) * c1 + source.dist * c2
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::PiQ, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    pow = @. exp(-process.β * process.r * t)
    c1 = @. (1 - pow) * process.π
    vsum = sum(source.dist .* c1, dims=1)
    dest.dist .= pow .* source.dist .+ vsum
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformDiscrete, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    scals = sum(source.dist, dims = 1)
    r = process.μ * 1/(1-1/K)   
    p = (1/K)
    pow = @. exp(-r * t)
    c1 = @. (1 - pow) * p
    c2 = @. pow + (1 - pow) * p
    dest.dist .= @. (scals - source.dist) * c1 + source.dist * c2
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformDiscrete, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    r = process.μ * 1/(1-1/K)   
    p = (1/K)
    pow = @. exp(-r * t)
    c1 = @. (1 - pow) * p
    vsum = sum(source.dist .* c1, dims=1)
    dest.dist .= pow .* source.dist .+ vsum
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformUnmasking, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    mask_volume = selectdim(source.dist, 1, K:K)
    event_p = @. 1 - exp(-process.μ * t)
    #Distribute lost mask volume among all other states equally, and decay it from the mask:
    selectdim(dest.dist, 1, 1:(K-1)) .= selectdim(source.dist, 1, 1:(K-1)) .+ mask_volume .* (1/(K-1)) .* event_p
    selectdim(dest.dist, 1, K:K) .= mask_volume .* (1 .- event_p)
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::UniformUnmasking, elapsed_time)
    t = expand(elapsed_time, ndims(source.dist))
    K = size(source.dist, 1)
    event_p = @. 1 - exp(-process.μ * t)
    #Nonmask states pass through unchanged.
    selectdim(dest.dist, 1, 1:(K-1)) .= selectdim(source.dist, 1, 1:(K-1))
    #Mask state's message gathers contributions from nonmask states.
    vsum = sum(selectdim(source.dist, 1, 1:(K-1)) .* (event_p/(K-1)), dims=1)
    selectdim(dest.dist, 1, K:K) .= (1 .- event_p) .* selectdim(source.dist, 1, K:K) .+ vsum
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::GeneralDiscrete, t::Real)
    P = exp(process.Q .* t)
    clamp!(P, 0, 1)
    reshape(dest.dist, size(source.dist,1), :) .= (reshape(source.dist, size(source.dist,1), :)' * P)'
    dest.log_norm_const .= source.log_norm_const
    return dest
end

function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::GeneralDiscrete, t::Real)
    P = exp(process.Q .* t)
    clamp!(P, 0, 1)
    mul!(reshape(dest.dist, size(source.dist,1), :), P, reshape(source.dist, size(source.dist,1), :))
    dest.log_norm_const .= source.log_norm_const
    return dest
end

#To add: DiagonalizadCTMC, HQtPi
#=
"""
    forward(process::HPiQ, x_s::AbstractArray, dt)

Computes the forward transition, supporting batch processing. The input `x_s` is an
AbstractArray where the first dimension corresponds to states, and subsequent
dimensions correspond to one or more batch dimensions.
"""
function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::HPiQ, dt)
    x_s = source.dist
    (; tree, π) = process
    N = length(π)
    @assert size(x_s, 1) == N "First dimension of x_s must match the number of states"
    
    no_event = ones(N)
    transitions = zeros(size(x_s))
    
    # Get all dimensions after the first (state) dimension
    data_dims = size(x_s)[2:end]
    
    # Helper to reshape a vector for broadcasting over the data dimensions
    reshape_for_broadcast(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))

    function forward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u
        p_no_event_node = exp.(-u * dt)
        p_event = 1.0 .- p_no_event_node
        p_no_event_ancestors = no_event[idx[1]]
        
        π_partition_view = view(π, idx)
        π_partition = π_partition_view / sum(π_partition_view)
        
        # Create a view into the partition and sum over the state dimension (1)
        x_s_partition_view = view(x_s, idx, ntuple(_ -> Colon(), length(data_dims))...)
        x_s_partition_sum = sum(x_s_partition_view, dims=1)
        
        # Calculate the transition term, broadcasting π over the data dimensions
        term = p_event .* p_no_event_ancestors * reshape_for_broadcast(π_partition) .* x_s_partition_sum
        
        transitions_view = view(transitions, idx, ntuple(_ -> Colon(), length(data_dims))...)
        transitions_view .+= term
        
        view(no_event, idx) .*= p_no_event_node
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    forward_recursive!(child)
                end
            end
        end
    end
    forward_recursive!(tree)
    dest.dist .= reshape_for_broadcast(no_event) .* x_s .+ transitions
    dest.log_norm_const .= source.log_norm_const
    return dest
end
=#

"""
    forward!(dest, source, process, dt)

Computes the forward transition, supporting batch processing with per-item time steps.

The input `x_s` is an AbstractArray where the first dimension corresponds to states,
and subsequent dimensions correspond to one or more batch dimensions.

The time step `dt` can be a scalar or an AbstractArray matching the batch dimensions
of `x_s`. For example, if `x_s` is a 3D tensor of size (num_states, height, width),
`dt` can be a matrix of size (height, width), allowing a different time step for
each column `x_s[:, i, j]`.
"""
#=
function forward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::HPiQ, dt::Union{Real, AbstractArray})
    x_s = source.dist
    (; tree, π) = process
    N = length(π)
    @assert size(x_s, 1) == N "First dimension of x_s must match the number of states"

    # Get all dimensions after the first (state) dimension
    data_dims = size(x_s)[2:end]

    # TODO
    # # Validate dt dimensions if it's an array
    # if dt isa AbstractArray
    #     @assert size(dt) == data_dims "Dimensions of dt must match the batch dimensions of x_s"
    # end
    
    # no_event now has the same dimensions as x_s to hold per-item probabilities
    no_event = ones(size(x_s))
    transitions = zeros(size(x_s))
    
    # Helper to reshape a state-dimension vector for broadcasting over the data dimensions
    reshape_state_for_broadcast(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))

    # Helper to reshape a data-dimension array for broadcasting over the state dimension (no-op for scalars)
    reshape_data_for_broadcast(a) = a isa AbstractArray ? reshape(a, (1, size(a)...)) : a

    function forward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u
        # p_no_event_node and p_event are now arrays with dimensions matching the batch dimensions
        p_no_event_node = exp.(-u .* dt) # Broadcasting dt
        p_event = 1.0 .- p_no_event_node
        
        # p_no_event_ancestors is a view into the no_event array.
        # Using a range idx[1]:idx[1] preserves the singleton dimension, making it a view
        # of size (1, data_dims...). This simplifies broadcasting later.
        p_no_event_ancestors = view(no_event, idx[1]:idx[1], ntuple(_ -> Colon(), length(data_dims))...)
        
        π_partition_view = view(π, idx)
        π_partition = π_partition_view / sum(π_partition_view)
        
        # Create a view into the partition and sum over the state dimension (1)
        x_s_partition_view = view(x_s, idx, ntuple(_ -> Colon(), length(data_dims))...)
        x_s_partition_sum = sum(x_s_partition_view, dims=1)
        
        # Calculate the transition term. Broadcasting happens across state and data dimensions.
        # We explicitly reshape p_event to avoid broadcasting ambiguity between arrays
        # of different dimensionality, which can cause DimensionMismatch errors.
        term = (reshape_data_for_broadcast(p_event) .* p_no_event_ancestors) .* reshape_state_for_broadcast(π_partition) .* x_s_partition_sum
        
        transitions_view = view(transitions, idx, ntuple(_ -> Colon(), length(data_dims))...)
        transitions_view .+= term
        
        # Update no_event probabilities for the states in the current partition.
        # p_no_event_node is broadcast over the states in idx.
        no_event_view = view(no_event, idx, ntuple(_ -> Colon(), length(data_dims))...)
        no_event_view .*= reshape_data_for_broadcast(p_no_event_node)
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    forward_recursive!(child)
                end
            end
        end
    end
    forward_recursive!(tree)
    
    # Final calculation: remaining probability mass + accumulated transitions
    dest.dist .= no_event .* x_s .+ transitions
    dest.log_norm_const .= source.log_norm_const
    return dest
end
=#

function forward!(
    dest::CategoricalLikelihood, 
    source::CategoricalLikelihood, 
    process::HPiQ, 
    dt::Union{Real, AbstractArray}
)
    x_s = source.dist
    (; tree, π) = process
    N = length(π)
    @assert size(x_s, 1) == N "First dimension of x_s must match the number of states"

    T=Float32
    # --- Pre-allocate all necessary workspace buffers ---
    data_dims = size(x_s)[2:end]
    # Use two separate, dedicated buffers to prevent memory overwriting bugs.
    p_no_event_buffer = similar(x_s, (data_dims...))
    p_event_buffer = similar(x_s, (data_dims...)) 
    sum_buffer = similar(x_s, (1, data_dims...))

    # Accumulate transitions directly into the destination array
    fill!(dest.dist, T(0.0))
    no_event = ones(size(x_s))
    
    reshape_state_for_broadcast(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))
    reshape_data_for_broadcast(a) = a isa AbstractArray ? reshape(a, (1, size(a)...)) : a

    function forward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u
        
        # --- Use dedicated buffers for each calculation ---
        # 1. Calculate p_no_event_node and p_event into their own buffers.
        p_no_event_buffer .= exp.(-u .* dt)
        p_event_buffer .= T(1.0) .- p_no_event_buffer

        # 2. Read the ANCESTOR survival probability. This is read BEFORE no_event is updated.
        p_no_event_ancestors = view(no_event, idx[1]:idx[1], ntuple(_ -> Colon(), length(data_dims))...)
        
        # 3. Calculate the transition term using the correct values.
        π_partition_view = view(π, idx)
        π_partition = π_partition_view / sum(π_partition_view)
        
        x_s_partition_view = view(x_s, idx, ntuple(_ -> Colon(), length(data_dims))...)
        sum!(sum_buffer, x_s_partition_view)
        x_s_partition_sum = sum_buffer
        
        term = (reshape_data_for_broadcast(p_event_buffer) .* p_no_event_ancestors) .* reshape_state_for_broadcast(π_partition) .* x_s_partition_sum
        
        dest_view = view(dest.dist, idx, ntuple(_ -> Colon(), length(data_dims))...)
        dest_view .+= term
        
        # 4. NOW, update the no_event array for the next level of recursion.
        no_event_view = view(no_event, idx, ntuple(_ -> Colon(), length(data_dims))...)
        no_event_view .*= reshape_data_for_broadcast(p_no_event_buffer)
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    forward_recursive!(child)
                end
            end
        end
    end
    
    forward_recursive!(tree)
    
    # Final update: dest.dist contains transitions, now add the no_event part.
    dest.dist .= no_event .* x_s .+ dest.dist
    
    dest.log_norm_const .= source.log_norm_const
    return dest
end

# function forward!(
#     dest::CategoricalLikelihood, 
#     source::CategoricalLikelihood, 
#     process::HPiQ, 
#     dt::Union{Real, AbstractArray};
#     # Note: dt_dim logic is removed for clarity, but can be added back in.
# )
#     x_s = source.dist
#     (; tree, π) = process
#     N = length(π)
#     @assert size(x_s, 1) == N "First dimension of x_s must match the number of states"

#     # --- OPTIMIZATION 1: Pre-allocate workspace buffers ---
#     # These buffers will be reused in every recursive call to avoid new allocations.
#     data_dims = size(x_s)[2:end]
#     # Buffer for p_no_event and p_event
#     p_buffer = similar(x_s, (data_dims...))
#     # Buffer for the sum over a partition
#     sum_buffer = similar(x_s, (1, data_dims...))

#     # --- OPTIMIZATION 2: Eliminate the `transitions` array ---
#     # We will accumulate transitions directly into the destination array.
#     fill!(dest.dist, 0)
#     # The no_event array is still needed as it's built recursively.
#     no_event = ones(size(x_s))
    
#     # Helper functions remain the same
#     reshape_state_for_broadcast(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))
#     reshape_data_for_broadcast(a) = a isa AbstractArray ? reshape(a, (1, size(a)...)) : a

#     function forward_recursive!(node::PiNode)
#         isnothing(node.leaf_indices) && return
#         idx = node.leaf_indices
#         isempty(idx) && return

#         u = node.u
        
#         # --- OPTIMIZATION 3: Use buffers for intermediate calculations ---
#         # Reuse p_buffer for p_no_event_node (no new allocation)
#         p_buffer .= exp.(-u .* dt)
#         p_no_event_node = p_buffer
        
#         p_no_event_ancestors = view(no_event, idx[1]:idx[1], ntuple(_ -> Colon(), length(data_dims))...)
        
#         π_partition_view = view(π, idx)
#         π_partition = π_partition_view / sum(π_partition_view)
        
#         x_s_partition_view = view(x_s, idx, ntuple(_ -> Colon(), length(data_dims))...)
#         # Use in-place sum! into sum_buffer (no new allocation)
#         sum!(sum_buffer, x_s_partition_view)
#         x_s_partition_sum = sum_buffer

#         # Reuse p_buffer for p_event (no new allocation)
#         p_buffer .= 1.0 .- p_no_event_node
#         p_event = p_buffer
        
#         # This line still allocates a temporary array for the result of the right-hand side,
#         # but we have eliminated the other major allocations within the recursion.
#         term = (reshape_data_for_broadcast(p_event) .* p_no_event_ancestors) .* reshape_state_for_broadcast(π_partition) .* x_s_partition_sum
        
#         # Accumulate directly into the destination array's view
#         dest_view = view(dest.dist, idx, ntuple(_ -> Colon(), length(data_dims))...)
#         dest_view .+= term
        
#         no_event_view = view(no_event, idx, ntuple(_ -> Colon(), length(data_dims))...)
#         no_event_view .*= reshape_data_for_broadcast(p_no_event_node)
        
#         if !isnothing(node.children)
#             for child in node.children
#                 if isa(child, PiNode)
#                     forward_recursive!(child)
#                 end
#             end
#         end
#     end
    
#     forward_recursive!(tree)
    
#     # --- OPTIMIZATION 4: Final update in-place ---
#     # `dest.dist` already contains the transitions. Now add the no_event part.
#     dest.dist .= no_event .* x_s .+ dest.dist
#     dest.log_norm_const .= source.log_norm_const
#     return dest
# end

"""
    backward!(dest, source, process, dt)

Computes the backward transition, supporting batch processing with per-item time steps.

The input `x_t` is an AbstractArray where the first dimension corresponds to states,
and subsequent dimensions correspond to one or more batch dimensions.

The time step `dt` can be a scalar or an AbstractArray matching the batch dimensions
of `x_t`. For example, if `x_t` is a 3D tensor of size (num_states, height, width),
`dt` can be a matrix of size (height, width), allowing a different time step for
each column `x_t[:, i, j]`.
"""
#=
function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::HPiQ, dt::Union{Real, AbstractArray})
    x_t = source.dist
    (; tree, π) = process
    N = length(π)
    @assert size(x_t, 1) == N "First dimension of x_t must match the number of states"

    # Get all dimensions after the first (state) dimension
    data_dims = size(x_t)[2:end]

    # # Validate dt dimensions if it's an array
    # if dt isa AbstractArray
    #     println(size(dt))
    #     println(size(data_dims))

    #     @assert size(dt) == data_dims "Dimensions of dt must match the batch dimensions of x_t"
    # end

    # no_event now has the same dimensions as x_t to hold per-item probabilities
    no_event = ones(size(x_t))
    transitions = zeros(size(x_t))
    
    # Helper to reshape a state-dimension vector for broadcasting over the data dimensions
    reshape_state_for_broadcast(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))

    # Helper to reshape a data-dimension array for broadcasting over the state dimension (no-op for scalars)
    reshape_data_for_broadcast(a) = a isa AbstractArray ? reshape(a, (1, size(a)...)) : a

    function backward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u
        # p_no_event_node and p_event are now arrays with dimensions matching the batch dimensions
        p_no_event_node = exp.(-u .* dt) # Broadcasting dt
        p_event = 1.0 .- p_no_event_node
        
        # p_no_event_ancestors is a view into the no_event array.
        # Using a range idx[1]:idx[1] preserves the singleton dimension, making it a view
        # of size (1, data_dims...).
        p_no_event_ancestors = view(no_event, idx[1]:idx[1], ntuple(_ -> Colon(), length(data_dims))...)
        
        π_partition_view = view(π, idx)
        π_partition = π_partition_view / sum(π_partition_view)
        
        # Create a view into the partition and calculate the weighted sum
        x_t_partition_view = view(x_t, idx, ntuple(_ -> Colon(), length(data_dims))...)
        x_t_partition_weighted_sum = sum(reshape_state_for_broadcast(π_partition) .* x_t_partition_view, dims=1)
        
        # Calculate the transition term with explicit reshaping to ensure correct broadcasting
        term = reshape_data_for_broadcast(p_event) .* p_no_event_ancestors .* x_t_partition_weighted_sum
        
        transitions_view = view(transitions, idx, ntuple(_ -> Colon(), length(data_dims))...)
        transitions_view .+= term
        
        # Update no_event probabilities for the states in the current partition
        no_event_view = view(no_event, idx, ntuple(_ -> Colon(), length(data_dims))...)
        no_event_view .*= reshape_data_for_broadcast(p_no_event_node)
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    backward_recursive!(child)
                end
            end
        end
    end

    backward_recursive!(tree)
    
    # Final calculation: remaining probability mass + accumulated transitions
    dest.dist .= no_event .* x_t .+ transitions
    dest.log_norm_const .= source.log_norm_const
    return dest
end
=#

function backward!(
    dest::CategoricalLikelihood, 
    source::CategoricalLikelihood, 
    process::HPiQ, 
    dt::Union{Real, AbstractArray}
)

    T = Float32
    x_t = source.dist
    (; tree, π) = process
    N = length(π)
    @assert size(x_t, 1) == N "First dimension of x_t must match the number of states"

    # --- Pre-allocate all necessary workspace buffers ---
    data_dims = size(x_t)[2:end]
    # Use two separate, dedicated buffers to prevent memory overwriting bugs.
    p_no_event_buffer = similar(x_t, (data_dims...))
    p_event_buffer = similar(x_t, (data_dims...)) 
    sum_buffer = similar(x_t, (1, data_dims...))

    # Accumulate transitions directly into the destination array
    # Use 0.0f0 for Float32 compatibility, or just 0.0 for Float64
    fill!(dest.dist, T(0.0)) 
    no_event = ones(size(x_t))
    
    reshape_state_for_broadcast(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))
    reshape_data_for_broadcast(a) = a isa AbstractArray ? reshape(a, (1, size(a)...)) : a

    function backward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u
        
        # --- Use dedicated buffers for each calculation ---
        # 1. Calculate p_no_event_node and p_event into their own buffers.
        p_no_event_buffer .= exp.(-u .* dt)
        p_event_buffer .= T(1.0) .- p_no_event_buffer # Use 1.0f0 for Float32

        # 2. Read the ANCESTOR survival probability. This is read BEFORE no_event is updated.
        p_no_event_ancestors = view(no_event, idx[1]:idx[1], ntuple(_ -> Colon(), length(data_dims))...)
        
        # 3. Calculate the transition term using the correct values.
        π_partition_view = view(π, idx)
        π_partition = π_partition_view / sum(π_partition_view)
        
        x_t_partition_view = view(x_t, idx, ntuple(_ -> Colon(), length(data_dims))...)
        
        # The element-wise product here still allocates a temporary array, but its size
        # is limited to the partition, not the full data dimension. The main memory
        # savings from buffering the large arrays are preserved.
        weighted_view = reshape_state_for_broadcast(π_partition) .* x_t_partition_view
        sum!(sum_buffer, weighted_view)
        x_t_partition_weighted_sum = sum_buffer
        
        term = reshape_data_for_broadcast(p_event_buffer) .* p_no_event_ancestors .* x_t_partition_weighted_sum
        
        dest_view = view(dest.dist, idx, ntuple(_ -> Colon(), length(data_dims))...)
        dest_view .+= term
        
        # 4. NOW, update the no_event array for the next level of recursion.
        no_event_view = view(no_event, idx, ntuple(_ -> Colon(), length(data_dims))...)
        no_event_view .*= reshape_data_for_broadcast(p_no_event_buffer)
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    backward_recursive!(child)
                end
            end
        end
    end

    backward_recursive!(tree)
    
    # Final update: dest.dist contains transitions, now add the no_event part.
    dest.dist .= no_event .* x_t .+ dest.dist
    
    dest.log_norm_const .= source.log_norm_const
    return dest
end


#=
"""
    backward(process::HPiQ, x_t::AbstractArray, s::Real, t::Real)

Computes the backward transition, supporting batch processing. The input `x_t` is an
AbstractArray where the first dimension corresponds to states, and subsequent
dimensions correspond to one or more batch dimensions.
"""
function backward!(dest::CategoricalLikelihood, source::CategoricalLikelihood, process::HPiQ, dt)
    x_t = source.dist
    (; tree, π) = process
    N = length(π)
    @assert size(x_t, 1) == N "First dimension of x_t must match the number of states"

    no_event = ones(N)
    transitions = zeros(size(x_t))
    
    data_dims = size(x_t)[2:end]
    reshape_for_broadcast(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))

    function backward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u
        p_no_event_node = exp(-u * dt)
        p_event = 1.0 - p_no_event_node
        p_no_event_ancestors = no_event[idx[1]]
        
        π_partition_view = view(π, idx)
        π_partition = π_partition_view / sum(π_partition_view)
        
        x_t_partition_view = view(x_t, idx, ntuple(_ -> Colon(), length(data_dims))...)
        x_t_partition_weighted_sum = sum(reshape_for_broadcast(π_partition) .* x_t_partition_view, dims=1)
        
        term = p_event * p_no_event_ancestors * x_t_partition_weighted_sum
        
        transitions_view = view(transitions, idx, ntuple(_ -> Colon(), length(data_dims))...)
        transitions_view .+= term
        
        view(no_event, idx) .*= p_no_event_node
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    backward_recursive!(child)
                end
            end
        end
    end

    backward_recursive!(tree)
    dest.dist .= reshape_for_broadcast(no_event) .* x_t .+ transitions
    dest.log_norm_const .= source.log_norm_const
    return dest
end
=#

