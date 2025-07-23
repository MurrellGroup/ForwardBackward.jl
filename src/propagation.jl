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

function forward!(
    dest::CategoricalLikelihood, 
    source::CategoricalLikelihood, 
    process::HPiQ, 
    elapsed_time::Union{Real, AbstractArray}
)
    dt = elapsed_time
    x_s = source.dist
    (; tree, π) = process
    T = eltype(π)
    N = length(π)
    @assert size(x_s, 1) == N "First dimension of x_s must match the number of states"
    data_dims = size(x_s)[2:end]
    p_no_event_buffer = similar(x_s, (data_dims...)) 
    p_event_buffer = similar(x_s, (data_dims...)) 
    sum_buffer = similar(x_s, (1, data_dims...))
    fill!(dest.dist, T(0.0))
    no_event = ones(size(x_s))
    
    expand_to_data_dims(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))
    expand_to_state_dim(a) = a isa AbstractArray ? reshape(a, (1, size(a)...)) : a

    function forward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u
        
        p_no_event_buffer .= exp.(-u .* dt) #(DDs...)
        p_event_buffer .= T(1.0) .- p_no_event_buffer #(DDs...)

        p_no_event_ancestors = view(no_event, idx[1]:idx[1], ntuple(_ -> Colon(), length(data_dims))...) #(1, DDs...)
        
        π_partition_view = view(π, idx) #(Leafs,)
        π_partition = π_partition_view / sum(π_partition_view) #(Leafs,)
        
        x_s_partition_view = view(x_s, idx, ntuple(_ -> Colon(), length(data_dims))...) #(Leafs, DDs)
        sum!(sum_buffer, x_s_partition_view)
        x_s_partition_sum = sum_buffer #(1, DDs...)
        
        # Brodcasting dimensions are: ( (1, DDs...) .*  (1, DDs) ) .* (Leafs, ntuple(Returns(1), length(data_dims))...) .* (1, DDs...)
        term = (expand_to_state_dim(p_event_buffer) .* p_no_event_ancestors) .* expand_to_data_dims(π_partition) .* x_s_partition_sum #(Leafs, DDs...)

        dest_view = view(dest.dist, idx, ntuple(_ -> Colon(), length(data_dims))...) #(Leafs, DDs...)
        dest_view .+= term #(Leafs, DDs...)
        
        no_event_view = view(no_event, idx, ntuple(_ -> Colon(), length(data_dims))...) #(Leafs, DDs...)
        no_event_view .*= expand_to_state_dim(p_no_event_buffer) #(Leafs, DDs..)
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    forward_recursive!(child)
                end
            end
        end
    end
    
    forward_recursive!(tree)
    dest.dist .= no_event .* x_s .+ dest.dist
    dest.log_norm_const .= source.log_norm_const
    
    return dest
end

function backward!(
    dest::CategoricalLikelihood, 
    source::CategoricalLikelihood, 
    process::HPiQ, 
    elapsed_time::Union{Real, AbstractArray}
)

    dt = elapsed_time
    x_t = source.dist
    (; tree, π) = process
    T = eltype(π)
    N = length(π)
    @assert size(x_t, 1) == N "First dimension of x_t must match the number of states"
    data_dims = size(x_t)[2:end]
    p_no_event_buffer = similar(x_t, (data_dims...)) 
    p_event_buffer = similar(x_t, (data_dims...))  
    sum_buffer = similar(x_t, (1, data_dims...)) 
    fill!(dest.dist, T(0.0)) 
    no_event = ones(size(x_t)) 
    
    expand_to_data_dims(v) = reshape(v, (length(v), ntuple(_ -> 1, length(data_dims))...))
    expand_to_state_dim(a) = a isa AbstractArray ? reshape(a, (1, size(a)...)) : a

    function backward_recursive!(node::PiNode)
        isnothing(node.leaf_indices) && return
        idx = node.leaf_indices
        isempty(idx) && return

        u = node.u

        p_no_event_buffer .= exp.(-u .* dt) #(DDs...)
        p_event_buffer .= T(1.0) .- p_no_event_buffer #(DDs...)

        p_no_event_ancestors = view(no_event, idx[1]:idx[1], ntuple(_ -> Colon(), length(data_dims))...) #(1, DDs...)
        
        π_partition_view = view(π, idx) #(Leafs,)
        π_partition = π_partition_view / sum(π_partition_view) #(Leafs,)
        
        x_t_partition_view = view(x_t, idx, ntuple(_ -> Colon(), length(data_dims))...) #(Leafs, DDs...)
        
        # Brodcasting dimensions are: (Leafs, ntuple(Returns(1), length(data_dims))...) .* (Leafs, DDs...)
        weighted_view = expand_to_data_dims(π_partition) .* x_t_partition_view #(Leafs, DDs...)
        sum!(sum_buffer, weighted_view)
        x_t_partition_weighted_sum = sum_buffer #(1, DDs...)
        
        # Brodcasting dimensions are: ( (1, DDs...) .*  (1, DDs...) ) .* (1, DDs...)
        term = expand_to_state_dim(p_event_buffer) .* p_no_event_ancestors .* x_t_partition_weighted_sum #(1, DDs...)
        
        dest_view = view(dest.dist, idx, ntuple(_ -> Colon(), length(data_dims))...) #(Leafs, DDs...)
        dest_view .+= term #(Leafs, DDs...)
        
        no_event_view = view(no_event, idx, ntuple(_ -> Colon(), length(data_dims))...)  #(Leafs, DDs...)
        no_event_view .*= expand_to_state_dim(p_no_event_buffer) #(Leafs, DDs...)
        
        if !isnothing(node.children)
            for child in node.children
                if isa(child, PiNode)
                    backward_recursive!(child)
                end
            end
        end
    end

    backward_recursive!(tree)
    dest.dist .= no_event .* x_t .+ dest.dist
    dest.log_norm_const .= source.log_norm_const
    
    return dest
end


#To add: DiagonalizadCTMC