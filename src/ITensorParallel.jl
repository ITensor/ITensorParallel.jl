module ITensorParallel

using Accessors
using Compat
using Distributed
using Folds
using ITensorMPS
using ITensors
using ITensors.NDTensors
using MPI

using ITensors: Algorithm, @Algorithm_str
using ITensorMPS: AbstractSum

import Base: eltype, length, size
import ITensors: product
import ITensorMPS:
  disk,
  linkind,
  lproj,
  noiseterm,
  nsite,
  orthogonalize!,
  position!,
  replaceind!,
  replacebond!,
  rproj,
  set_terms,
  terms
include("partition/partition.jl")
include("partition/partition_sum_split.jl")
include("partition/partition_chain_split.jl")
include("force_gc.jl")
include("foldssum.jl")
include("distributedsum.jl")
include("mpi_extensions.jl")
include("mpisumterm.jl")

export DistributedSum, SequentialSum, ThreadedSum, MPISumTerm, distribute, partition

end
