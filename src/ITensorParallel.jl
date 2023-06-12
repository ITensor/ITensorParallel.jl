module ITensorParallel

using Accessors
using Distributed
using Folds
using MPI
using ITensors
using ITensors.NDTensors

using ITensors: AbstractSum, Algorithm, @Algorithm_str

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

include(joinpath("partition", "partition.jl"))
include(joinpath("partition", "partition_sum_split.jl"))
include(joinpath("partition", "partition_chain_split.jl"))
include("foldssum.jl")
include("distributedsum.jl")
include("mpi_extensions.jl")
include("mpisumterm.jl")

export DistributedSum, SequentialSum, ThreadedSum, MPISumTerm, distribute, partition

end
