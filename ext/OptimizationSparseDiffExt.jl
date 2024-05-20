module OptimizationSparseDiffExt

import OptimizationBase, OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.ADTypes: AutoSparse, AutoFiniteDiff, AutoForwardDiff,
                                 AutoReverseDiff
using OptimizationBase.LinearAlgebra, ReverseDiff
isdefined(Base, :get_extension) ?
(using SparseDiffTools,
       SparseDiffTools.ForwardDiff, SparseDiffTools.FiniteDiff, Symbolics) :
(using ..SparseDiffTools,
       ..SparseDiffTools.ForwardDiff, ..SparseDiffTools.FiniteDiff, ..Symbolics)

function default_chunk_size(len)
    if len < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        len
    else
        ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    end
end

include("OptimizationSparseForwardDiff.jl")

const FD = FiniteDiff

include("OptimizationSparseFiniteDiff.jl")

struct OptimizationSparseReverseTag end

include("OptimizationSparseReverseDiff.jl")
end
