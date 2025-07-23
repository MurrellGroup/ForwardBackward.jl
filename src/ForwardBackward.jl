module ForwardBackward

using Distributions, LinearAlgebra, ArraysOfArrays

include("maths.jl")
include("processes.jl")
include("states.jl")
include("propagation.jl")
include("manifolds.jl")

export
    #Abstract Types
    Process,
    DiscreteProcess,
    State,
    StateLikelihood,
    #Processes
    Deterministic,    
    BrownianMotion,
    OrnsteinUhlenbeck,
    UniformDiscrete,
    UniformUnmasking,
    GeneralDiscrete,
    PiQ,
    HPiQ,
    PiNode,
    PiLeaf,
    #Likelihoods & States
    CategoricalLikelihood,
    GaussianLikelihood,
    DiscreteState,
    ContinuousState,
    #Functions
    endpoint_conditioned_sample,
    interpolate,
    ⊙,
    forward,
    backward,
    forward!,
    backward!,
    tensor,
    sumnorm,
    stochastic,
    init_leaf_indices!,
    add_child!,
    #Manifolds
    ManifoldProcess,
    ManifoldState,
    perturb!,
    perturb,
    expand
    
end