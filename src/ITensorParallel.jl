module ITensorParallel

using Accessors
using Distributed
using Folds
using MPI
using ITensors
using ITensors.NDTensors

using ITensors: AbstractSum

import Base: eltype, length, size
import ITensors:
  disk,
  linkind,
  lproj,
  noiseterm,
  nsite,
  orthogonalize!,
  position!,
  product,
  replaceind!,
  replacebond!,
  rproj,
  set_terms,
  terms

include("partition.jl")
include("foldssum.jl")
include("distributedsum.jl")
include("mpi_extensions.jl")
include("mpisumterm.jl")

export DistributedSum, SequentialSum, ThreadedSum, MPISumTerm, distribute, partition

end
