var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ForwardBackward","category":"page"},{"location":"#ForwardBackward.jl","page":"Home","title":"ForwardBackward.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ForwardBackward.jl is a Julia package for evolving discrete and continuous states under a variety of processes.","category":"page"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package implements forward and backward methods for a number of useful processes. For times s  t  u:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Xt = forward(Xs, P, t-s) computes the distribution propto P(X_t  X_s)\nXt = backward(Xu, P, u-t) computes the likelihood propto P(X_u  X_t) (ie. considered as a function of X_t)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Where P is a Process, and each of X_s, X_t, X_u can be DiscreteState or ContinuousState, or scaled distributions (CategoricalLikelihood or GaussianLikelihood) over states, where the uncertainty (and normalizing constants) propogate. States and Likelihoods hold arrays of points/distributions, which are all acted upon by the process. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"Since CategoricalLikelihood and GaussianLikelihood are closed under elementwise/pointwise products, to compute the (scaled) distribution at t, which is P(X_t  X_s X_u)  P(X_t  X_s)P(X_u  X_t), we also provide ⊙:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Xt = forward(Xs, P, t-s) ⊙ backward(Xu, P, u-t)","category":"page"},{"location":"","page":"Home","title":"Home","text":"One use-cases is drawing endpoint conditioned samples:","category":"page"},{"location":"","page":"Home","title":"Home","text":"rand(forward(Xs, P, t-s) ⊙ backward(Xu, P, u-t))","category":"page"},{"location":"","page":"Home","title":"Home","text":"or","category":"page"},{"location":"","page":"Home","title":"Home","text":"endpoint_conditioned_sample(Xs, Xu, P, t-s, u-t)","category":"page"},{"location":"","page":"Home","title":"Home","text":"For some processes where we don't support propagation of uncertainty, (eg. the ManifoldProcess), endpoint_conditioned_sample is implemented directly via approximate simulation.","category":"page"},{"location":"#Processes","page":"Home","title":"Processes","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Continuous State\nBrownianMotion\nOrnsteinUhlenbeck\nDeterministic (where endpoint_conditioned_sample interpolates)\nDiscrete State:\nGeneralDiscrete, with any Q matrix, where propogation is via matrix exponentials\nUniformDiscrete, with all rates equal\nPiQ, where any event is a switch to a draw from the stationary distribution\nUniformUnmasking, where switches occur from a masked state to any other states","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"ForwardBackward\")","category":"page"},{"location":"#Quick-Start","page":"Home","title":"Quick Start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using ForwardBackward\n\n# Create a Brownian motion process\nprocess = BrownianMotion(0.0, 1.0)  # drift = 0.0, variance = 1.0\n\n# Define start and end states\nX0 = ContinuousState(zeros(10))     # start at origin\nX1 = ContinuousState(ones(10))      # end at ones\n\n# Sample a path at t = 0.3\nsample = endpoint_conditioned_sample(X0, X1, process, 0.3)","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"#Discrete-State-Process","page":"Home","title":"Discrete State Process","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"# Create a process with uniform transition rates\nprocess = UniformDiscrete()\nX0 = DiscreteState(4, [1])    # 4 possible states, starting in state 1\nX1 = DiscreteState(4, [4])    # ending in state 4\n\n# Sample intermediate state\nsample = endpoint_conditioned_sample(X0, X1, process, 0.5)","category":"page"},{"location":"#Manifold-Valued-Process","page":"Home","title":"Manifold-Valued Process","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Manifolds\n\n# Create a process on a sphere\nM = Sphere(2)                  # 2-sphere\nprocess = ManifoldProcess(0.1) # with some noise\n\n# Define start and end points\np0 = [1.0, 0.0, 0.0]\np1 = [0.0, 0.0, 1.0]\nX0 = ManifoldState(M, [p0])\nX1 = ManifoldState(M, [p1])\n\n# Sample a path\nsample = endpoint_conditioned_sample(X0, X1, process, 0.5)","category":"page"},{"location":"#Endpoint-conditioned-samples-on-a-torus","page":"Home","title":"Endpoint-conditioned samples on a torus","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"#Project Torus(2) into 3D (just for plotting)\nfunction tor(p; R::Real=2, r::Real=0.5)\n    u,v = p[1], p[2]\n    x = (R + r*cos(u)) * cos(v)\n    y = (R + r*cos(u)) * sin(v)\n    z = r * sin(u)\n    return [x, y, z]\nend\n\n#When non-zero, the process will diffuse. When 0, the process is deterministic:\nfor P in [ManifoldProcess(0), ManifoldProcess(0.05)]\n    #When non-zero, the endpoints will be slightly noised:\n    for perturb_var in [0.0, 0.0001] \n     \n        #Define the manifold, and two endpoints, which are on opposite sides (in both dims) of the torus:\n        M = ForwardBackward.Torus(2)\n        p1 = [-pi, 0.0]\n        p0 = [0.0, -pi]\n\n        #We'll generate endpoint-conditioned samples evenly spaced over time:\n        t_vec = 0:0.001:1\n\n        #Set up the X0 and X1 states, just repeating the endpoints over and over:\n        X0 = ManifoldState(M, [perturb(M, p0, perturb_var) for _ in t_vec])\n        X1 = ManifoldState(M, [perturb(M, p1, perturb_var) for _ in t_vec])\n\n        #Independently draw endpoint-conditioned samples at times t_vec:\n        Xt = endpoint_conditioned_sample(X0, X1, P, t_vec)\n\n        #Plot the torus, and the endpoint conditioned samples upon it:\n        R = 2\n        r = 0.5\n        u = range(0, 2π; length=100)  # angle around the tube\n        v = range(0, 2π; length=100)  # angle around the torus center\n        pl = plot([(R + r*cos(θ))*cos(φ) for θ in u, φ in v], [(R + r*cos(θ))*sin(φ) for θ in u, φ in v], [r*sin(θ) for θ in u, φ in v],\n            color = \"grey\", alpha = 0.3, label = :none, camera = (30,30))\n\n        #Map the points to 3D and plot them:\n        endpts = stack(tor.([p0,p1]))\n        smppts = stack(tor.(eachcol(tensor(Xt))))\n        scatter!(smppts[1,:], smppts[2,:], smppts[3,:], label = :none, msw = 0, ms = 1.5, color = \"blue\", alpha = 0.5)\n        scatter!(endpts[1,:], endpts[2,:], endpts[3,:], label = :none, msw = 0, ms = 2.5, color = \"red\")\n        savefig(\"torus_$(perturb_var)_$(P.v).svg\")\n    end\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Image) (Image: Image) (Image: Image) (Image: Image)","category":"page"},{"location":"#API-Reference","page":"Home","title":"API Reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ForwardBackward]","category":"page"},{"location":"#ForwardBackward.BrownianMotion","page":"Home","title":"ForwardBackward.BrownianMotion","text":"BrownianMotion(δ::Real, v::Real)\nBrownianMotion()\n\nBrownian motion process with drift δ and variance v.\n\nParameters\n\nδ: Drift parameter (default: 0.0)\nv: Variance parameter (default: 1.0)\n\nExamples\n\n# Standard Brownian motion\nprocess = BrownianMotion()\n\n# Brownian motion with drift 0.5 and variance 2.0\nprocess = BrownianMotion(0.5, 2.0)\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.CategoricalLikelihood","page":"Home","title":"ForwardBackward.CategoricalLikelihood","text":"CategoricalLikelihood(dist::AbstractArray, log_norm_const::AbstractArray)\nCategoricalLikelihood(K::Int, dims...; T=Float64)\nCategoricalLikelihood(dist::AbstractArray)\n\nProbability distribution over discrete states.\n\nParameters\n\ndist: Probability masses for each state\nlog_norm_const: Log normalization constants\nK: Number of categories (for initialization)\ndims: Additional dimensions for initialization\nT: Numeric type (default: Float64)\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.ContinuousProcess","page":"Home","title":"ForwardBackward.ContinuousProcess","text":"abstract type ContinuousProcess <: Process end\n\nBase type for processes with continuous state spaces.\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.ContinuousState","page":"Home","title":"ForwardBackward.ContinuousState","text":"ContinuousState(state::AbstractArray{<:Real})\n\nRepresentation of continuous states.\n\nParameters\n\nstate: Array of current state values\n\nExamples\n\n# Create a continuous state\nstate = ContinuousState(randn(100))\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.Deterministic","page":"Home","title":"ForwardBackward.Deterministic","text":"Deterministic()\n\nA deterministic process where endpoint conditioning results in linear interpolation between states.\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.DiscreteProcess","page":"Home","title":"ForwardBackward.DiscreteProcess","text":"abstract type DiscreteProcess <: Process end\n\nBase type for processes with discrete state spaces.\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.GaussianLikelihood","page":"Home","title":"ForwardBackward.GaussianLikelihood","text":"GaussianLikelihood(mu::AbstractArray, var::AbstractArray, log_norm_const::AbstractArray)\n\nGaussian probability distribution over continuous states.\n\nParameters\n\nmu: Mean values\nvar: Variances\nlog_norm_const: Log normalization constants\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.GeneralDiscrete","page":"Home","title":"ForwardBackward.GeneralDiscrete","text":"GeneralDiscrete(Q::Matrix)\n\nDiscrete process with arbitrary transition rate matrix Q.\n\nParameters\n\nQ: Transition rate matrix\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.ManifoldProcess","page":"Home","title":"ForwardBackward.ManifoldProcess","text":"ManifoldProcess(v::T)\nManifoldProcess()\n\nA stochastic process on a Riemannian manifold with drift variance v.\n\nParameters\n\nv: Drift variance parameter (default: 0)\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.ManifoldState","page":"Home","title":"ForwardBackward.ManifoldState","text":"ManifoldState{Q<:AbstractManifold,A<:AbstractArray}(state::A, M::Q)\nManifoldState(M::AbstractManifold, state::AbstractArray{<:AbstractArray})\n\nRepresents a state on a Riemannian manifold.\n\nParameters\n\nstate: Array of points on the manifold\nM: The manifold object\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.OrnsteinUhlenbeck","page":"Home","title":"ForwardBackward.OrnsteinUhlenbeck","text":"OrnsteinUhlenbeck(μ::Real, v::Real, θ::Real)\nOrnsteinUhlenbeck()\n\nOrnstein-Uhlenbeck process with mean μ, variance v, and mean reversion rate θ.\n\nParameters\n\nμ: Long-term mean (default: 0.0)\nv: Variance parameter (default: 1.0)\nθ: Mean reversion rate (default: 1.0)\n\nExamples\n\n# Standard OU process\nprocess = OrnsteinUhlenbeck()\n\n# OU process with custom parameters\nprocess = OrnsteinUhlenbeck(1.0, 0.5, 2.0)\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.PiQ","page":"Home","title":"ForwardBackward.PiQ","text":"PiQ(r::Real, π::Vector{<:Real}; normalize=true)\nPiQ(π::Vector{<:Real}; normalize=true)\n\nDiscrete process that switches to states proportionally to the stationary distribution π with rate r.\n\nParameters\n\nr: Overall switching rate (default: 1.0)\nπ: Target stationary distribution (will always be normalized to sum to 1)\nnormalize: Whether to normalize the expected substitutions per unit time to be 1 when r = 1 (default: true)\n\nExamples\n\n# Process with uniform stationary distribution\nprocess = PiQ(ones(4) ./ 4)\n\n# Process with custom stationary distribution and rate\nprocess = PiQ(2.0, [0.1, 0.2, 0.3, 0.4])\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.Process","page":"Home","title":"ForwardBackward.Process","text":"abstract type Process end\n\nBase type for all stochastic processes in the package.\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.State","page":"Home","title":"ForwardBackward.State","text":"abstract type State end\n\nBase type for all state representations.\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.StateLikelihood","page":"Home","title":"ForwardBackward.StateLikelihood","text":"abstract type StateLikelihood end\n\nBase type for probability distributions over states.\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.UniformDiscrete","page":"Home","title":"ForwardBackward.UniformDiscrete","text":"UniformDiscrete(μ::Real)\nUniformDiscrete()\n\nDiscrete process with uniform transition rates between states, scaled by μ.\n\nParameters\n\nμ: Rate scaling parameter (default: 1.0)\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.UniformUnmasking","page":"Home","title":"ForwardBackward.UniformUnmasking","text":"UniformUnmasking(μ::Real)\nUniformUnmasking()\n\nMutates only a mask (the last state index) to any other state (with equal rates). When everything is masked,  μ=1 corresponds to one substitution per unit time.\n\nParameters\n\nμ: Rate parameter (default: 1.0)\n\n\n\n\n\n","category":"type"},{"location":"#ForwardBackward.:⊙-Tuple{CategoricalLikelihood, CategoricalLikelihood}","page":"Home","title":"ForwardBackward.:⊙","text":"⊙(a::CategoricalLikelihood, b::CategoricalLikelihood; norm=true)\n⊙(a::GaussianLikelihood, b::GaussianLikelihood)\n\nCompute the pointwise product of two likelihood distributions. For Gaussian likelihoods, this results in another Gaussian. For categorical likelihoods, this results in another categorical distribution.\n\nParameters\n\na, b: Input likelihood distributions\nnorm: Whether to normalize the result (categorical only, default: true)\n\nReturns\n\nA new likelihood distribution of the same type as the inputs.\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.backward!-Tuple{ForwardBackward.StateLikelihood, ForwardBackward.State, ForwardBackward.Process, Any}","page":"Home","title":"ForwardBackward.backward!","text":"backward!(Xdest::StateLikelihood, Xt::State, process::Process, t)\nbackward(Xt::StateLikelihood, process::Process, t)\nbackward(Xt::State, process::Process, t)\n\nPropagate a state or likelihood backward in time according to the process dynamics.\n\nParameters\n\nXdest: Destination for in-place operation\nXt: Final state or likelihood\nprocess: The stochastic process\nt: Time to propagate backward\n\nReturns\n\nThe backward-propagated state or likelihood\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.endpoint_conditioned_sample-NTuple{5, Any}","page":"Home","title":"ForwardBackward.endpoint_conditioned_sample","text":"endpoint_conditioned_sample(X0, X1, p, tF, tB)\nendpoint_conditioned_sample(X0, X1, p, t)\nendpoint_conditioned_sample(X0, X1, p::Deterministic, tF, tB)\n\nGenerate a sample from the endpoint-conditioned process.\n\nParameters\n\nX0: Initial state\nX1: Final state\np: The stochastic process\nt, tF: Forward time\ntB: Backward time (defaults to 1-t for single time parameter)\n\nReturns\n\nA sample from the endpoint-conditioned distribution\n\nNotes\n\nFor continuous processes, uses the forward-backward algorithm. For deterministic processes, uses linear interpolation.\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.endpoint_conditioned_sample-Tuple{ManifoldState, ManifoldState, ManifoldProcess, Any, Any}","page":"Home","title":"ForwardBackward.endpoint_conditioned_sample","text":"endpoint_conditioned_sample(X0::ManifoldState, X1::ManifoldState, p::ManifoldProcess, tF, tB; Δt = 0.05)\n\nGenerate a sample from the endpoint-conditioned process on a manifold.\n\nParameters\n\nX0: Initial state\nX1: Final state\np: The manifold process\ntF: Forward time\ntB: Backward time\nΔt: Discretized step size (default: 0.05)\n\nReturns\n\nA sample state at the specified time\n\nNotes\n\nUses a numerical stepping procedure to approximate the endpoint-conditioned distribution.\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.forward!-Tuple{ForwardBackward.StateLikelihood, ForwardBackward.State, ForwardBackward.Process, Any}","page":"Home","title":"ForwardBackward.forward!","text":"forward!(Xdest::StateLikelihood, Xt::State, process::Process, t)\nforward(Xt::StateLikelihood, process::Process, t)\nforward(Xt::State, process::Process, t)\n\nPropagate a state or likelihood forward in time according to the process dynamics.\n\nParameters\n\nXdest: Destination for in-place operation\nXt: Initial state or likelihood\nprocess: The stochastic process\nt: Time to propagate forward\n\nReturns\n\nThe forward-propagated state or likelihood\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.interpolate!-Tuple{ManifoldState, ManifoldState, ManifoldState, Any, Any}","page":"Home","title":"ForwardBackward.interpolate!","text":"interpolate!(dest::ManifoldState, X0::ManifoldState, Xt::ManifoldState, tF, tB)\ninterpolate(X0::ManifoldState, Xt::ManifoldState, tF, tB)\n\nInterpolate between two states on a manifold using geodesics.\n\nParameters\n\ndest: Destination state for in-place operation\nX0: Initial state\nXt: Final state\ntF: Time difference from initial state\ntB: Time difference to final state\n\nReturns\n\nThe interpolated state\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.interpolate-Tuple{ContinuousState, ContinuousState, Any, Any}","page":"Home","title":"ForwardBackward.interpolate","text":"interpolate(X0::ContinuousState, X1::ContinuousState, tF, tB)\n\nLinearly interpolate between two continuous states.\n\nParameters\n\nX0: Initial state\nX1: Final state\ntF: Forward time\ntB: Backward time\n\nReturns\n\nThe interpolated state\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.perturb!-Tuple{ManifoldsBase.AbstractManifold, Any, Any, Any}","page":"Home","title":"ForwardBackward.perturb!","text":"perturb!(M::AbstractManifold, q, p, v)\nperturb(M::AbstractManifold, p, v)\n\nPerturb a point p on manifold M by sampling from a normal distribution in the tangent space with variance v and exponentiating back to the manifold.\n\nParameters\n\nM: The manifold\nq: The point that is overwritten (for perturb!)\np: Original point\nv: Variance of perturbation\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.perturb-Tuple{ManifoldsBase.AbstractManifold, Any, Any}","page":"Home","title":"ForwardBackward.perturb","text":"perturb!(M::AbstractManifold, q, p, v)\nperturb(M::AbstractManifold, p, v)\n\nPerturb a point p on manifold M by sampling from a normal distribution in the tangent space with variance v and exponentiating back to the manifold.\n\nParameters\n\nM: The manifold\nq: The point that is overwritten (for perturb!)\np: Original point\nv: Variance of perturbation\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.step_toward!-Tuple{ManifoldsBase.AbstractManifold, Vararg{Any, 6}}","page":"Home","title":"ForwardBackward.step_toward!","text":"step_toward!(M::AbstractManifold, dest, p, q, var, delta_t, remaining_t)\nstep_toward(M::AbstractManifold, p, q, var, delta_t, remaining_t)\n\nTake a single diffusion step from point p toward point q on manifold M. If var is 0, this is a deterministic step along the geodesic.\n\nParameters\n\nM: The manifold\ndest: Destination for the new point\np: Starting point\nq: Target point\nvar: Variance of stochastic perturbation\ndelta_t: Time step size\nremaining_t: Total remaining time\n\nReturns\n\nThe new point after stepping\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.stochastic-Tuple{Type, ContinuousState}","page":"Home","title":"ForwardBackward.stochastic","text":"stochastic(o::State)\nstochastic(T::Type, o::State)\n\nConvert a state to its corresponding likelihood distribution: A zero-variance (ie. delta function) Gaussian for the continuous case, and a one-hot categorical distribution for the discrete case.\n\nParameters\n\no: Input state\nT: Numeric type for the resulting distribution (default: Float64)\n\nReturns\n\nA likelihood distribution corresponding to the input state.\n\n\n\n\n\n","category":"method"},{"location":"#ForwardBackward.tensor-Tuple{ForwardBackward.State}","page":"Home","title":"ForwardBackward.tensor","text":"tensor(d::Union{State, StateLikelihood})\n\nConvert a state or likelihood to its tensor (ie. multidimensional array) representation.\n\nReturns\n\nThe underlying array representation of the state or likelihood.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"```","category":"page"}]
}
