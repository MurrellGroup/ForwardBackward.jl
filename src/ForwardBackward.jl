module ForwardBackward

    using Distributions, LinearAlgebra, Manifolds, ArraysOfArrays

    include("maths.jl")
    include("processes.jl")
    include("states.jl")
    include("propogation.jl")
    include("manifolds.jl")

    export
        #Processes
        Deterministic,    
        BrownianMotion,
        OrnsteinUhlenbeck,
        UniformDiscrete,
        GeneralDiscrete,
        PiQ,
        ManifoldProcess,
        #Likelihoods & States
        CategoricalLikelihood,
        GaussianLikelihood,
        DiscreteState,
        ContinuousState,
        ManifoldState,
        #Functions
        sumnorm,
        endpoint_conditioned_sample,
        interpolate,
        ⊙,
        forward,
        backward,
        forward!,
        backward!,
        tensor
    
end