struct FoldsSum{T,Ex} <: AbstractSum
  terms::Vector{T}
  executor::Ex
end

## Accessors
terms(sum::FoldsSum) = sum.terms
executor(sum::FoldsSum) = sum.executor
set_terms(sum::FoldsSum, terms) = (@set sum.terms = terms)
set_executor(sum::FoldsSum, executor) = (@set sum.executor = executor)

## Constructors
function FoldsSum{T,Ex}(terms::Vector; executor_kwargs...) where {T,Ex}
  return FoldsSum(terms, Ex(; executor_kwargs...))
end

function FoldsSum{<:Any,Ex}(terms::Vector; executor_kwargs...) where {Ex}
  return FoldsSum{eltype(terms),Ex}(terms; executor_kwargs...)
end

# Default to `ThreadedEx`
function FoldsSum{T}(terms::Vector; executor_kwargs...) where {T}
  return FoldsSum{T,ThreadedEx}(terms; executor_kwargs...)
end

# Default to `ThreadedEx`
function FoldsSum(terms::Vector; executor_kwargs...)
  return FoldsSum{eltype(terms)}(terms; executor_kwargs...)
end

# Conversion from `MPO`
function FoldsSum{T,Ex}(mpos::Vector{MPO}; executor_kwargs...) where {T,Ex}
  return FoldsSum{T,Ex}(T.(mpos); executor_kwargs...)
end
function FoldsSum{T,Ex}(mpos::MPO...; executor_kwargs...) where {T,Ex}
  return FoldsSum{T,Ex}([mpos...]; executor_kwargs...)
end

# Conversion from `MPO`
function FoldsSum{T}(mpos::Vector{MPO}; executor_kwargs...) where {T}
  return FoldsSum{T}(T.(mpos); executor_kwargs...)
end
function FoldsSum{T}(mpos::MPO...; executor_kwargs...) where {T}
  return FoldsSum{T}([mpos...]; executor_kwargs...)
end

# Conversion from `MPO`, default to `ProjMPO`
function FoldsSum{<:Any,Ex}(mpos::Vector{MPO}; executor_kwargs...) where {Ex}
  return FoldsSum{ProjMPO,Ex}(mpos; executor_kwargs...)
end
function FoldsSum{<:Any,Ex}(mpos::MPO...; executor_kwargs...) where {Ex}
  return FoldsSum{<:Any,T}([mpos...]; executor_kwargs...)
end

# Conversion from `MPO`, default to `ProjMPO`
function FoldsSum(mpos::Vector{MPO}; executor_kwargs...)
  return FoldsSum{ProjMPO}(mpos; executor_kwargs...)
end
function FoldsSum(mpos::MPO...; executor_kwargs...)
  return FoldsSum([mpos...]; executor_kwargs...)
end

## Necessary operations
function product(sum::FoldsSum, v::ITensor)
  return Folds.sum(term -> term(v), terms(sum), executor(sum))
end

function position!(sum::FoldsSum, v::MPS, pos::Int)
  new_terms = Folds.map(term -> position!(term, v, pos), terms(sum), executor(sum))
  return set_terms(sum, new_terms)
end

function noiseterm(sum::FoldsSum, v::ITensor, dir::String)
  return Folds.sum(term -> noiseterm(term, v, dir), terms(sum), executor(sum))
end

const ThreadedSum{T} = FoldsSum{T,ThreadedEx}
const SequentialSum{T} = FoldsSum{T,SequentialEx}
