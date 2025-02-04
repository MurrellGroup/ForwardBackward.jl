```@meta
CurrentModule = ForwardBackward
```

# ForwardBackward.jl

ForwardBackward.jl is a Julia package for endpoint-conditioned sampling and interpolation of stochastic processes. It provides a framework for generating paths between two states that respect the underlying dynamics of various processes.

## Overview

The package allows you to:
1. Define different types of stochastic processes (continuous and discrete)
2. Represent states and their likelihoods
3. Perform forward and backward propagation of distributions
4. Generate endpoint-conditioned samples between two states

## Core Concepts

### Processes

The package supports several types of processes:

- **Continuous State Processes**
  - `BrownianMotion`: Standard Brownian motion with drift `δ` and variance `v`
  - `OrnsteinUhlenbeck`: OU process with mean `μ`, variance `v`, and mean reversion rate `θ`

- **Discrete State Processes**
  - `UniformDiscrete`: Uniform switching rates between states
  - `UniformUnmasking`: A process that unmasks states
  - `PiQ`: A switching event that switches to each state proportionally to the stationary distribution
  - `GeneralDiscrete`: A process with arbitrary transition rate matrix
  
- **Manifold Processes**
  - `ManifoldProcess`: A process on a manifold with drift variance `v`

### States and Likelihoods

A `State` is a collection of values (with flexible dimensionality).States can be either discrete, continuous, or points on a manifold:

```julia
state = DiscreteState(4, rand(1:4, 100))  # 100 states with 4 possible values

state = ContinuousState(randn(100))  # 100 continuous values

M = ForwardBackward.Sphere(2)
state = ManifoldState(M, rand(M, 100))  # 100 points on a sphere
```

A `Likelihood` is a distribution and a log-normalization constant. `DiscreteState`s have a `CategoricalLikelihood` representation, and `ContinuousState`s have a `GaussianLikelihood` representation. These support the propogation of uncertainty under the processes. `ManifoldState`s do not have a likelihood representation, and endpoint-conditioned sampling is done via a approximate simulation.

### Endpoint-Conditioned Sampling

Endpoint-conditioned sampling is the draw of a sample from a process that is conditioned to start and end at specified states (or state likelihoods, where supported). This is achieved through:

1. Forward propagation from the initial state
2. Backward propagation from the final state
3. Combining the likelihoods using the pointwise product (⊙)
4. Sampling from the combined likelihood

```julia
# Example for continuous process
X0 = ContinuousState(zeros(10))  # Initial state
X1 = ContinuousState(ones(10))   # Final state
process = BrownianMotion()  # Standard Brownian motion

# Generate a sample at t=0.3 given endpoints at t=0 and t=1
t = 0.3
sample = endpoint_conditioned_sample(X0, X1, process, t)
```

## Mathematical Background

The endpoint-conditioned sampling works by exploiting the fact that:

P(Xt | X0, X1) ∝ P(Xt | X0) × P(X1 | Xt)

where:
- P(Xt | X0), considered as a function of Xt, is computed by forward propagation
- P(X1 | Xt), considered as a function of Xt, is computed by backward propagation
- × represents pointwise multiplication of likelihoods.

## Usage Examples

```julia
using ForwardBackward

# Brownian Motion example
process = BrownianMotion(0.0, 1.0)
X0 = ContinuousState(zeros(10))
X1 = ContinuousState(ones(10))
t = 0.3

# Forward propagation
forward_dist = forward(X0, process, t)

# Backward propagation
backward_dist = backward(X1, process, 1-t)

# Combine distributions and sample
sample = rand(forward_dist ⊙ backward_dist)

# Or use the convenience function
sample = endpoint_conditioned_sample(X0, X1, process, t)
```

## API Reference

```@index
```

```@autodocs
Modules = [ForwardBackward]
```
```