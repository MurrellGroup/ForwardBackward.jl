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
    OrnsteinUhlenbeckExpVar,
    UniformDiscrete,
    UniformUnmasking,
    GeneralDiscrete,
    PiQ,
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
    #Manifolds
    ManifoldProcess,
    ManifoldState,
    perturb!,
    perturb,
    expand
    
end