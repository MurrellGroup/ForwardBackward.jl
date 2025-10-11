using ManifoldsBase
import Functors

"""
    ManifoldProcess(v::T)

A stochastic process on a Riemannian manifold with drift variance `v`.

# Parameters
- `v`: Drift variance parameter (default: 0)
"""
struct ManifoldProcess{T} <: Process
    v::T
end

ManifoldProcess() = ManifoldProcess(0)

"""
    ManifoldState(M::AbstractManifold, state::AbstractArray{<:AbstractArray})

Represents a state on a Riemannian manifold.

# Parameters
- `state`: Array of points on the manifold
- `M`: The manifold object

---

    ManifoldState(M::ProbabilitySimplex, x::AbstractArray{<:Integer}; softner! = ForwardBackward.soften!)

Convert a discrete array to points on a probability simplex. `maximum(x)` must be `<= manifold_dimension(M)+1`.
By default this moves the points slightly away from the corners of the simplex (see `soften!`).
"""
struct ManifoldState{Q<:AbstractManifold,A<:AbstractArray} <: State
    M::Q
    state::A
end

function ManifoldState(M::AbstractManifold, state::AbstractArray{<:AbstractArray})
    @invoke ManifoldState(M, ArrayOfSimilarArrays(state)::AbstractArray)
end

Base.similar(S::ManifoldState) = ManifoldState(S.M, similar(S.state))
Base.copy(S::ManifoldState) = ManifoldState(S.M, copy(S.state))

Functors.@functor ManifoldState (state,)

function Functors.functor(::Type{<:ManifoldState{<:AbstractManifold,<:ArrayOfSimilarArrays}}, state)
    namedtuple = (; data=flatview(state.state))
    reconstruct = nt -> @invoke ManifoldState(state.M, nestedview(nt.data, length(innersize(state.state)))::AbstractArray)
    return namedtuple, reconstruct
end


function interpolate!(dest::ManifoldState, X0::ManifoldState, Xt::ManifoldState, tF, tB)
    shortest_geodesic!.((X0.M,), dest.state, X0.state, Xt.state, expand(tF ./ (tF .+ tB), ndims(X0.state)))
    return dest
end

"""
    interpolate(X0::ManifoldState, Xt::ManifoldState, tF, tB)

Interpolate between two states on a manifold using geodesics.

# Parameters
- `dest`: Destination state for in-place operation
- `X0`: Initial state
- `Xt`: Final state
- `tF`: Time difference from initial state
- `tB`: Time difference to final state

# Returns
The interpolated state
"""
interpolate(X0::ManifoldState, Xt::ManifoldState, tF, tB) = interpolate!(similar(X0), X0, Xt, tF, tB)


MultiNormal(d::Int, v::Real) = MvNormal(ForwardBackward.LinearAlgebra.Diagonal(Distributions.FillArrays.Fill(v, d)))

"""
    perturb!(M::AbstractManifold, q, p, v)

Perturb a point `p` on manifold `M` by sampling from a normal distribution in the tangent space
with variance `v` and exponentiating back to the manifold.

# Parameters
- `M`: The manifold
- `q`: The point that is overwritten (for perturb!)
- `p`: Original point
- `v`: Variance of perturbation
"""
perturb!(M::AbstractManifold, q, p, v) = exp!(M, q, p, get_vector(M, p, rand(MultiNormal(manifold_dimension(M), v))))

"""
    perturb(M::AbstractManifold, p, v)

Non-mutating version of [`perturb!`](@ref).
"""
perturb(M::AbstractManifold, p, v) = perturb!(M, similar(p), p, v)


# A bit uncertain about this for manifolds. Might need to apply the drift correction in the tangent space.
# Take a single diffusion step from point `p` toward point `q` on manifold `M`. If `var` is 0, this is a deterministic step along the geodesic.
function step_toward!(M::AbstractManifold, dest, p, q, var::Real, t_a, t_b, t_c)
    delta_t = t_b .- t_a
    remaining_t = t_c .- t_b
    if remaining_t > 0
        new_p = (var > 0) ? perturb!(M, dest, p, delta_t * var) : p
        shortest_geodesic!(M, dest, new_p, q, delta_t / remaining_t)
    end
    return dest
end


# This could be a lot faster. Reducing allocations, etc.
function step_toward!(M::AbstractManifold, dest, p, q, base_process::Process, t_a, t_b, t_c)
    q_tan_coords = get_coordinates(M,p,log(M,p,q))
    p_coords = zero(q_tan_coords)
    t_b_coords = endpoint_conditioned_sample(ContinuousState(p_coords), ContinuousState(q_tan_coords), base_process, t_a, t_b, t_c)
    exp!(M, p, dest, get_vector(M, p, tensor(t_b_coords)))
    return dest
end


step_toward(M::AbstractManifold, p, q, var, t_a, t_b, t_c) = step_toward!(M, similar(p), p, q, var, t_a, t_b, t_c)


"""
    endpoint_conditioned_sample(X0::ManifoldState, X1::ManifoldState, p::ManifoldProcess, tF, tB; Δt = 0.05)

Generate a sample from the endpoint-conditioned process on a manifold.

# Parameters
- `X0`: Initial state
- `X1`: Final state
- `p`: The manifold process
- `tF`: Forward time
- `tB`: Backward time
- `Δt`: Discretized step size (default: 0.05)

# Returns
A sample state at the specified time

# Notes
Uses a numerical stepping procedure to approximate the endpoint-conditioned distribution.
"""
function endpoint_conditioned_sample(X0::ManifoldState, X1::ManifoldState, p::ManifoldProcess, t_a, t_b, t_c; Δt = 0.05)
    T = eltype(flatview(X0.state))
    t_a_arr = zeros(T, size(X0.state)) .+ expand(t_a, ndims(X0.state))
    t_b_arr = zeros(T, size(X0.state)) .+ expand(t_b, ndims(X0.state))
    t_c_arr = zeros(T, size(X0.state)) .+ expand(t_c, ndims(X0.state))
    Xt = copy(X0)
    for ind in CartesianIndices(X0.state)
        t = t_a_arr[ind]
        while t < t_b_arr[ind]
            inc = min(T(t+Δt), t_b_arr[ind])
            step_toward!(X0.M, Xt.state[ind], Xt.state[ind], X1.state[ind], p.v, t, inc, t_c_arr[ind])
            t = inc
        end
    end
    return Xt
end

#You should only call this for time-homogeneous processes:
endpoint_conditioned_sample(X0::ManifoldState, X1::ManifoldState, p::ManifoldProcess, tF, tB; Δt = 0.05) = endpoint_conditioned_sample(X0, X1, p, zero(tF), tF, tF + tB, Δt = Δt)


# ManifoldsExt
"""
    soften!(x::AbstractArray{T}, a = T(1e-5)) where T

In-place regularizes values of `x` slightly while preserving its sum along the first dimension.
```
(x .+ a) ./ sum(x .+ a, dims = 1)
```
"""
function soften!(x::AbstractArray{T}, a = T(1e-5)) where T
    x .= x .+ a
    x .= x ./ sum(x, dims = 1)
end


#=
#These work, but we might not need them. Need to think about it.

function perturb!(D::ManifoldState, S::ManifoldState, v)
    perturb!.((S.M,), D.state, S.state, expand(v, ndims(S.state)))
    return D
end
perturb(S::ManifoldState, v) = perturb!(similar(S), S, v)
=#


#=
M = Sphere(2)
p0 = rand(M)
p1 = rand(M)
X0 = ManifoldState(M, [p0 for _ in zeros(10, 101)])
X1 = ManifoldState(M, [p1 for _ in zeros(10, 101)])
@time Xt = endpoint_conditioned_sample(X0, X1, ManifoldProcess(0.1), collect(0:0.01:1))
fv = flatview(Xt.state)
pl = plot()
for i in 1:1000
    b = rand(M)
    scatter!([b[1]], [b[2]], [b[3]], alpha = 0.1, label = :none, msw = 0, color = "red")
end
for i in 1:10
    scatter!(fv[1,i,:], fv[2,i,:], fv[3,i,:], alpha = 0.1, label = :none, msw = 0, color = "blue")
end
pl
=#


#=
M = Torus(2)
p0 = rand(M)
p1 = rand(M)
X0 = ManifoldState(M, [p0 for _ in zeros(10, 101)])
X1 = ManifoldState(M, [p1 for _ in zeros(10, 101)])
@time Xt = endpoint_conditioned_sample(X0, X1, ManifoldProcess(0.1), collect(0:0.02:2), 2 .- collect(0:0.02:2))
pl = plot()
for i in 1:10
    scatter!(flatview(Xt.state)[1,i,:], flatview(Xt.state)[2,i,:], alpha = 0.1, label = :none, msw = 0, color = "blue")
end
pl
=#

#=
#A 1D Brownian bridge. Full stepping trajectories vs endpoint conditioned samples (blue points).
#The mean of all the trajectories should converge to the geodesic between the two points.
M = Euclidean(1)
p_init = [-1.0]
q = [1.0]
p_init, q
curves = []
pl = plot()
ts = nothing
@time for i in 1:500
    p = copy(p_init)
    var = 0.5
    delta_t = 0.01
    T = 1.0
    t = 0.0
    ts = Float64[t]
    points = [p[1]]
    while t < T
        step = min(delta_t, T-t)
        p = step_toward(M, p, q, var, step, T-t)
        t += step
        push!(points, p[1])
        push!(ts, t)
    end
    plot!(ts, points, alpha = 0.3, label = :none)
    push!(curves, points)
end
plot!(ts,mean(curves), color = "red", label = "Mean")
plot!([0,1],[p_init[1],q[1]], color = "black", linestyle = :dash, label = "Geodesic")
pl

M = Euclidean(1)
X0 = ManifoldState(M, [[-1.0] for _ in zeros(50, 101)])
X1 = ManifoldState(M, [[1.0] for _ in zeros(50, 101)])
@time Xt = endpoint_conditioned_sample(X0, X1, ManifoldProcess(0.5), collect(0:0.01:1), 1 .- collect(0:0.01:1))
for i in 1:50
    scatter!(collect(0:0.01:1), flatview(Xt.state)[1,i,:], alpha = 0.1, label = :none, msw = 0, color = "blue")
end
pl
=#



