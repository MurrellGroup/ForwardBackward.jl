# ForwardBackward.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/ForwardBackward.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/ForwardBackward.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/ForwardBackward.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ForwardBackward.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/ForwardBackward.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/ForwardBackward.jl)

Some helpful functions for some simple stochastic processes.

## Overview

The package implements `forward` and `backward` methods for a number of useful processes. For times $s < t < u$:
- `Xt = forward(Xs, P, t-s)` computes the distribution $\propto P(X_t | X_s)$
- `Xt = backward(Xu, P, u-t)` computes the likelihood $\propto P(X_u | X_t)$ (ie. considered as a function of $X_t$)

Where `P` is a `Process`, and each of $X_s$, $X_t$, $X_u$ can be `DiscreteState` or `ContinuousState`, or scaled distributions (`CategoricalLikelihood` or `GaussianLikelihood`) over states, where the uncertainty (and normalizing constants) propogate. `States` and `Likelihoods` hold arrays of points/distributions, which are all acted upon by the process. 

Since `CategoricalLikelihood` and `GaussianLikelihood` are closed under elementwise/pointwise products, to compute the (scaled) distribution at $t$, which is $P(X_t | X_s, X_u) ∝ P(X_t | X_s)P(X_u | X_t)$, we also provide `⊙`:

```julia
Xt = forward(Xs, P, t-s) ⊙ backward(Xu, P, u-t)
```

One use-cases is drawing endpoint conditioned samples:
```julia
rand(forward(Xs, P, t-s) ⊙ backward(Xu, P, u-t))
```
or
```julia
endpoint_conditioned_sample(Xs, Xu, P, t-s, u-t)
```

For some processes where we don't support propagation of uncertainty, (eg. the `ManifoldProcess`), `endpoint_conditioned_sample` is implemented directly via approximate simulation.

## Processes
- **Continuous State**
  - `BrownianMotion`
  - `OrnsteinUhlenbeck`
  - `Deterministic` (where `endpoint_conditioned_sample` interpolates)
  
- **Discrete State**:
  - `GeneralDiscrete`, with any `Q` matrix, where propogation is via matrix exponentials
  - `UniformDiscrete`, with all rates equal
  - `PiQ`, where any event is a switch to a draw from the stationary distribution
  - `UniformUnmasking`, where switches occur from a masked state to any other states

## Installation

```julia
using Pkg
Pkg.add("ForwardBackward")
```

## Quick Start

```julia
using ForwardBackward

# Create a Brownian motion process
process = BrownianMotion(0.0, 1.0)  # drift = 0.0, variance = 1.0

# Define start and end states
X0 = ContinuousState(zeros(10))     # start at origin
X1 = ContinuousState(ones(10))      # end at ones

# Sample a path at t = 0.3
sample = endpoint_conditioned_sample(X0, X1, process, 0.3)
```

## Examples

### Discrete State Process
```julia
# Create a process with uniform transition rates
process = UniformDiscrete()
X0 = DiscreteState(4, [1])    # 4 possible states, starting in state 1
X1 = DiscreteState(4, [4])    # ending in state 4

# Sample intermediate state
sample = endpoint_conditioned_sample(X0, X1, process, 0.5)
```

### Manifold-Valued Process
```julia
using Manifolds

# Create a process on a sphere
M = Sphere(2)                  # 2-sphere
process = ManifoldProcess(0.1) # with some noise

# Define start and end points
p0 = [1.0, 0.0, 0.0]
p1 = [0.0, 0.0, 1.0]
X0 = ManifoldState(M, [p0])
X1 = ManifoldState(M, [p1])

# Sample a path
sample = endpoint_conditioned_sample(X0, X1, process, 0.5)
```

### Endpoint-conditioned samples on a torus

```julia
using ForwardBackward, Manifolds, Plots

#Project Torus(2) into 3D (just for plotting)
function tor(p; R::Real=2, r::Real=0.5)
    u,v = p[1], p[2]
    x = (R + r*cos(u)) * cos(v)
    y = (R + r*cos(u)) * sin(v)
    z = r * sin(u)
    return [x, y, z]
end

#Define the manifold, and two endpoints, which are on opposite sides (in both dims) of the torus:
M = Torus(2)
p1 = [-pi, 0.0]
p0 = [0.0, -pi]

#When non-zero, the process will diffuse. When 0, the process is deterministic:
for P in [ManifoldProcess(0), ManifoldProcess(0.05)]
    #When non-zero, the endpoints will be slightly noised:
    for perturb_var in [0.0, 0.0001] 
     
        #We'll generate endpoint-conditioned samples evenly spaced over time:
        t_vec = 0:0.001:1

        #Set up the X0 and X1 states, just repeating the endpoints over and over:
        X0 = ManifoldState(M, [perturb(M, p0, perturb_var) for _ in t_vec])
        X1 = ManifoldState(M, [perturb(M, p1, perturb_var) for _ in t_vec])

        #Independently draw endpoint-conditioned samples at times t_vec:
        Xt = endpoint_conditioned_sample(X0, X1, P, t_vec)

        #Plot the torus:
        R = 2
        r = 0.5
        u = range(0, 2π; length=100)  # angle around the tube
        v = range(0, 2π; length=100)  # angle around the torus center
        pl = plot([(R + r*cos(θ))*cos(φ) for θ in u, φ in v], [(R + r*cos(θ))*sin(φ) for θ in u, φ in v], [r*sin(θ) for θ in u, φ in v],
            color = "grey", alpha = 0.3, label = :none, camera = (30,30))

        #Map the points to 3D and plot them:
        endpts = stack(tor.([p0,p1]))
        smppts = stack(tor.(eachcol(tensor(Xt))))
        scatter!(smppts[1,:], smppts[2,:], smppts[3,:], label = :none, msw = 0, ms = 1.5, color = "blue", alpha = 0.5)
        scatter!(endpts[1,:], endpts[2,:], endpts[3,:], label = :none, msw = 0, ms = 2.5, color = "red")
        savefig("torus_$(perturb_var)_$(P.v).svg")
    end
end
```

`torus_0.0_0.svg:`

![Image](https://github.com/user-attachments/assets/21410c12-fd16-4542-b323-5f048e878bb5)

`torus_0.0001_0.svg:`

![Image](https://github.com/user-attachments/assets/a88d67a1-87f6-44a2-9b70-2315c3eaa983)

`torus_0.0_0.05.svg:`

![Image](https://github.com/user-attachments/assets/fb3dc348-3fcf-4a3c-b120-521db0a9350d)

`torus_0.0001_0.05.svg:`

![Image](https://github.com/user-attachments/assets/06e65a05-cc3d-4cfb-95cc-d6c27b0211c7)

## License

This project is licensed under the MIT License - see the LICENSE file for details.