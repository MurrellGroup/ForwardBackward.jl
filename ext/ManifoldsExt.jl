module ManifoldsExt

using ForwardBackward, Manifolds, ManifoldsBase

# By default, this moves the points slightly away from the corners of the simplex, for... reasons.
function ForwardBackward.ManifoldState(
    T::Type,
    M::ProbabilitySimplex{ManifoldsBase.TypeParameter{Tuple{K}}, :open},
    x::AbstractArray{<:Integer};
    softner! = ForwardBackward.soften!
) where K
    s = ManifoldState(M, eachslice(stochastic(T, DiscreteState(K+1, x)).dist, dims = Tuple(1 .+ collect(1:ndims(x)))))
    !isnothing(softner!) && softner!(tensor(s))
    return s
end

ForwardBackward.ManifoldState(M::ProbabilitySimplex, x::AbstractArray{<:Integer}; kws...) = ManifoldState(Float64, M, x; kws...)


end