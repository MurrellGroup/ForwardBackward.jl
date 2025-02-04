"""
    ManifoldProcess(v::T)

A process on a manifold, with a drift variance `v`.
"""
struct ManifoldProcess{T} <: Process
    v::T
end

ManifoldProcess() = ManifoldProcess(0)

struct ManifoldState{Q<:AbstractManifold,A<:AbstractArray} <: State
    state::A
    M::Q
end

ManifoldState(M::AbstractManifold, state::AbstractArray{<:AbstractArray}) = ManifoldState(ArrayOfSimilarArrays(state), M)

Base.similar(S::ManifoldState) = ManifoldState(S.M, similar(S.state))
Base.copy(S::ManifoldState) = ManifoldState(S.M, copy(S.state))

function interpolate!(dest::ManifoldState, X0::ManifoldState, Xt::ManifoldState, Δt0, Δt1)
    shortest_geodesic!.((X0.M,), dest.state, X0.state, Xt.state, expand(Δt0 ./ (Δt0 .+ Δt1), ndims(X0.state)))
    return dest
end
interpolate(X0::ManifoldState, Xt::ManifoldState, Δt0, Δt1) = interpolate!(similar(X0), X0, Xt, Δt0, Δt1)

"""
    perturb(M, p, v)

Perturb a point on a manifold by a normal distribution with variance `v`.
"""
perturb!(M::AbstractManifold, q, p, v) = exp!(M, q, p, get_vector(M, p, rand(MvNormal(manifold_dimension(M), sqrt(v)))))

#A bit uncertain about this for manifolds. Might need to apply the drift correction in the tangent space.
function step_toward!(M::AbstractManifold, dest, p, q, var, delta_t, remaining_t)
    new_p = (var > 0) ? perturb!(M, dest, p, delta_t * var) : p
    shortest_geodesic!(M, dest, new_p, q, delta_t / remaining_t)
    return dest
end
step_toward(M::AbstractManifold, p, q, var, delta_t, remaining_t) = step_toward!(M, similar(p), p, q, var, delta_t, remaining_t)

function endpoint_conditioned_sample(X0::ManifoldState, X1::ManifoldState, p::ManifoldProcess, tF, tB; Δt = 0.05)
    T = eltype(flatview(X0.state))
    tot = zeros(T, size(X0.state)) .+ expand(tF .+ tB, ndims(X0.state))
    target = zeros(T, size(X0.state)) .+ expand(tF, ndims(X0.state))
    Xt = copy(X0)
    for ind in CartesianIndices(X0.state)
        t = 0.0
        while t < target[ind]
            inc = min(T(Δt), target[ind] - t)
            step_toward!(X0.M, Xt.state[ind], Xt.state[ind], X1.state[ind], p.v, inc, tot[ind] - t)
            t += inc
        end
    end
    return Xt
end


#=
#These work, but we might not need them. Need to think about it.
perturb(M::AbstractManifold, p, v) = perturb!(M, similar(p), p, v)
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



