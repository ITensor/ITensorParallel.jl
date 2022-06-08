struct ThreadedProjMPOSum
  pm::Vector{ProjMPO}
end

ThreadedProjMPOSum(mpos::Vector{MPO}) = ThreadedProjMPOSum([ProjMPO(M) for M in mpos])

ThreadedProjMPOSum(Ms::MPO...) = ThreadedProjMPOSum([Ms...])

nsite(P::ThreadedProjMPOSum) = nsite(P.pm[1])

Base.length(P::ThreadedProjMPOSum) = length(P.pm[1])

function product(P::ThreadedProjMPOSum, v::ITensor)::ITensor
  Pvs = fill(ITensor(), Threads.nthreads())
  Threads.@threads for M in P.pm
    Pvs[Threads.threadid()] += product(M, v)
  end
  return sum(Pvs)
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
  Threads.@threads for M in P.pm
    position!(M, psi, pos)
  end
  return P
end

function noiseterm(P::ThreadedProjMPOSum, phi::ITensor, dir::String)
  nts = fill(ITensor(), Threads.nthreads())
  Threads.@threads for M in P.pm
    nts[Threads.threadid()] += noiseterm(M, phi, dir)
  end
  return sum(nts)
end
