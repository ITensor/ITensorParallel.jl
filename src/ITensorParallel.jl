module ITensorParallel

using Distributed
using Folds
using MPI
using ITensors
using ITensors.NDTensors

import Base: eltype, length, size
import ITensors: product, position!, noiseterm, lproj, rproj, nsite, replaceind!, linkind

include("mpi_extensions.jl")
include("partition.jl")
include("threadedsum.jl")
include("mpisum.jl")

# BROKEN
# include("distributedsum.jl")

export partition,
  ThreadedSum,
  ThreadedProjMPOSum,
  MPISum

end
