struct ThreadedProjMPOSum
  pm::Vector{ProjMPO}
end

ThreadedProjMPOSum(mpos::Vector{MPO}) = ThreadedProjMPOSum([ProjMPO(M) for M in mpos])

ThreadedProjMPOSum(Ms::MPO...) = ThreadedProjMPOSum([Ms...])

nsite(P::ThreadedProjMPOSum) = nsite(P.pm[1])

Base.length(P::ThreadedProjMPOSum) = length(P.pm[1])

function product(P::ThreadedProjMPOSum, v::ITensor)
  return Folds.sum(M -> product(M, v), P.pm, ThreadedEx())
end

function Base.eltype(P::ThreadedProjMPOSum)
  elT = eltype(P.pm[1])
  for n in 2:length(P.pm)
    elT = promote_type(elT, eltype(P.pm[n]))
  end
  return elT
end

(P::ThreadedProjMPOSum)(v::ITensor) = product(P, v)

Base.size(P::ThreadedProjMPOSum) = size(P.pm[1])

function position!(P::ThreadedProjMPOSum, psi::MPS, pos::Int)
  Folds.map(M -> position!(M, psi, pos), P.pm, ThreadedEx())
  return P
end

function noiseterm(P::ThreadedProjMPOSum, phi::ITensor, dir::String)
  return Folds.sum(M -> noiseterm(M, phi, dir), P.pm, ThreadedEx())
end
