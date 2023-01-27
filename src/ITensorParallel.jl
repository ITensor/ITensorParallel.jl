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
include("parallelsum.jl")
include("mpisum.jl")

export DistributedSum, SequentialSum, ThreadedSum, MPISum, partition

end
