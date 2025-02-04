module ForwardBackward

    using Distributions, LinearAlgebra, Manifolds, ArraysOfArrays

    include("maths.jl")
    include("processes.jl")
    include("states.jl")
    include("propagation.jl")
    include("manifolds.jl")

    export
        #Processes
        Deterministic,    
        BrownianMotion,
        OrnsteinUhlenbeck,
        UniformDiscrete,
        UniformUnmasking,
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
        perturb!,
        perturb
    
end