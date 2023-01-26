struct ParallelSum{T,Ex}
  terms::Vector{T}
  executor::Ex
end

terms(sum::ParallelSum) = sum.terms
executor(sum::ParallelSum) = sum.executor
set_terms(sum::ParallelSum, terms) = ParallelSum(terms, executor(sum))

function ParallelSum{T,Ex}(terms::Vector; executor_kwargs...) where {T,Ex}
  return ParallelSum(terms, Ex(; executor_kwargs...))
end

function ParallelSum{<:Any,Ex}(terms::Vector; executor_kwargs...) where {Ex}
  return ParallelSum{eltype(terms),Ex}(terms; executor_kwargs...)
end

# Default to `ThreadedEx`
function ParallelSum{T}(terms::Vector; executor_kwargs...) where {T}
  return ParallelSum{T,ThreadedEx}(terms; executor_kwargs...)
end

# Default to `ThreadedEx`
function ParallelSum(terms::Vector; executor_kwargs...)
  return ParallelSum{eltype(terms)}(terms; executor_kwargs...)
end

# Conversion from `MPO`
function ParallelSum{T,Ex}(mpos::Vector{MPO}; executor_kwargs...) where {T,Ex}
  return ParallelSum{T,Ex}(T.(mpos); executor_kwargs...)
end
function ParallelSum{T,Ex}(mpos::MPO...; executor_kwargs...) where {T,Ex}
  return ParallelSum{T,Ex}([mpos...]; executor_kwargs...)
end

# Conversion from `MPO`
function ParallelSum{T}(mpos::Vector{MPO}; executor_kwargs...) where {T}
  return ParallelSum{T}(T.(mpos); executor_kwargs...)
end
function ParallelSum{T}(mpos::MPO...; executor_kwargs...) where {T}
  return ParallelSum{T}([mpos...]; executor_kwargs...)
end

# Conversion from `MPO`, default to `ProjMPO`
function ParallelSum{<:Any,Ex}(mpos::Vector{MPO}; executor_kwargs...) where {Ex}
  return ParallelSum{ProjMPO,Ex}(mpos; executor_kwargs...)
end
function ParallelSum{<:Any,Ex}(mpos::MPO...; executor_kwargs...) where {Ex}
  return ParallelSum{<:Any,T}([mpos...]; executor_kwargs...)
end

# Conversion from `MPO`, default to `ProjMPO`
function ParallelSum(mpos::Vector{MPO}; executor_kwargs...)
  return ParallelSum{ProjMPO}(mpos; executor_kwargs...)
end
function ParallelSum(mpos::MPO...; executor_kwargs...)
  return ParallelSum([mpos...]; executor_kwargs...)
end

# Replace with `AbstractSum`
# https://github.com/ITensor/ITensors.jl/pull/1046
nsite(sum::ParallelSum) = nsite(terms(sum)[1])
length(sum::ParallelSum) = length(terms(sum)[1])
function eltype(sum::ParallelSum)
  elT = eltype(terms(sum)[1])
  for n in 2:length(terms(sum))
    elT = promote_type(elT, eltype(terms(sum)[n]))
  end
  return elT
end
(sum::ParallelSum)(v::ITensor) = product(sum, v)
size(sum::ParallelSum) = size(terms(sum)[1])

function product(sum::ParallelSum, v::ITensor)
  return Folds.sum(term -> product(term, v), terms(sum), executor(sum))
end

function position!(sum::ParallelSum, psi::MPS, pos::Int)
  exec = executor(sum)
  if exec isa DistributedEx
    # Use threaded instead
    exec = ThreadedEx()
  end
  Folds.map!(term -> position!(term, psi, pos), terms(sum), terms(sum), exec)
  return sum
end

function noiseterm(sum::ParallelSum, phi::ITensor, dir::String)
  return Folds.sum(term -> noiseterm(term, phi, dir), terms(sum), executor(sum))
end

function disk(sum::ParallelSum; disk_kwargs...)
  return set_terms(sum, [disk(term; disk_kwargs...) for term in terms(sum)])
end

const ThreadedSum{T} = ParallelSum{T,ThreadedEx}
const DistributedSum{T} = ParallelSum{T,DistributedEx}
const SequentialSum{T} = ParallelSum{T,SequentialEx}

const ThreadedProjMPOSum = ThreadedSum{ProjMPO}
const DistributedProjMPOSum = DistributedSum{ProjMPO}
const SequentialProjMPOSum = SequentialSum{ProjMPO}
