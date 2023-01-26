struct ThreadedSum{T}
  pm::Vector{T}
end

ThreadedSum{T}(mpos::Vector{MPO}) where {T} = ThreadedSum([T(M) for M in mpos])
ThreadedSum{T}(Ms::MPO...) where {T} = ThreadedSum{T}([Ms...])

# Default to `ProjMPO`
ThreadedSum(mpos::Vector{MPO}) = ThreadedSum{ProjMPO}(mpos)
ThreadedSum(Ms::MPO...) = ThreadedSum([Ms...])

nsite(P::ThreadedSum) = nsite(P.pm[1])

length(P::ThreadedSum) = length(P.pm[1])

function product(P::ThreadedSum, v::ITensor)
  return Folds.sum(M -> product(M, v), P.pm, ThreadedEx())
end

function eltype(P::ThreadedSum)
  elT = eltype(P.pm[1])
  for n in 2:length(P.pm)
    elT = promote_type(elT, eltype(P.pm[n]))
  end
  return elT
end

(P::ThreadedSum)(v::ITensor) = product(P, v)

size(P::ThreadedSum) = size(P.pm[1])

function position!(P::ThreadedSum, psi::MPS, pos::Int)
  Folds.map(M -> position!(M, psi, pos), P.pm, ThreadedEx())
  return P
end

function noiseterm(P::ThreadedSum, phi::ITensor, dir::String)
  return Folds.sum(M -> noiseterm(M, phi, dir), P.pm, ThreadedEx())
end

function disk(P::ThreadedSum; disk_kwargs...)
  return ThreadedSum([disk(M; disk_kwargs...) for M in P.pm])
end

const ThreadedProjMPOSum = ThreadedSum{ProjMPO}
