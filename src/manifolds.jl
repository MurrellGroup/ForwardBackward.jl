
struct ManifoldProcess{T} <: Process
    v::T
end

struct ManifoldState{Q,A<:AbstractArray} <: State
    state::A
    M::Q
end

""" Load `using Manifolds` to use ManifoldProcess """
function ManifoldProcess end
""" Load `using Manifolds` to use ManifoldState """
function ManifoldState end
""" Load `using Manifolds` to use perturb! """
function perturb! end
""" Load `using Manifolds` to use perturb """
function perturb end

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



