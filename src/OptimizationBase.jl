module OptimizationBase

using DocStringExtensions
using Reexport
@reexport using SciMLBase, ADTypes

if !isdefined(Base, :get_extension)
    using Requires
end

using ArrayInterface, Base.Iterators, SparseArrays, LinearAlgebra, StaticArraysCore

import SciMLBase: OptimizationProblem,
    OptimizationFunction, ObjSense,
    MaxSense, MinSense, OptimizationStats
export ObjSense, MaxSense, MinSense

include("adtypes.jl")
include("cache.jl")
include("function.jl")

export solve, OptimizationCache

end
