module ITensorParallel

using Distributed
using Folds
using MPI
using ITensors
using ITensors.NDTensors

import ITensors: product, position!, noiseterm, lproj, rproj, nsite, replaceind!, linkind

include("partition.jl")
include("threaded_projmposum.jl")
include("distributed_projmposum.jl")
include("mpi_aux.jl")
include("mpi_projmposum.jl")

export ThreadedProjMPOSum, partition, DistributedSum, MPISum

end
