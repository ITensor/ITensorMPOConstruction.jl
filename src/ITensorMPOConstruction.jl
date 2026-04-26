module ITensorMPOConstruction

#####################################
# External packages
#
using ITensorMPS
using ITensors
using Dictionaries
using LinearAlgebra
using SparseArrays
using TimerOutputs
using FunctionWrappers
using Printf
using ThreadsX

#####################################
# MPO Construction
#
include("time_if.jl")
include("OpIDSum.jl")
include("ops.jl")
include("BipartiteGraph.jl")
include("connected-components.jl")
include("minimum-vertex-cover.jl")
include("large-graph-mpo-common.jl")
include("large-graph-mpo-qr.jl")
include("large-graph-mpo-vertex-cover.jl")
include("sparse_tensor_construction.jl")
include("MPOConstruction.jl")

#####################################
# Exports
#

# OpIDSum.jl
export OpInfo, OpCacheVec, to_OpCacheVec, OpID, OpIDSum

# MPOConstruction.jl
export resume_MPO_construction!, MPO_new, instantiate_MPO, sparsity, block2_nnz

end
