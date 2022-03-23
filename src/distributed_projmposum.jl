struct DistributedSum{T}
  pm::Vector{T}
end

function DistributedSum(f::Function, n::Integer)
  # Assign the matrices to the first `n` workers.
  return DistributedSum([@spawnat(workers()[ni], f(ni)) for ni in 1:n])
end

nsite(P::DistributedSum) = nsite(fetch(P.pm[1]))

Base.length(P::DistributedSum) = length(fetch(P.pm[1]))

function product(P::DistributedSum, v::ITensor)::ITensor
  return @distributed (+) for M in P.pm
    fetch(M)(v)
  end
end

function Base.eltype(P::DistributedSum)
  elT = eltype(P.pm[1])
  for n=2:length(P.pm)
    elT = promote_type(elT,eltype(P.pm[n]))
  end
  return elT
end

(P::DistributedSum)(v::ITensor) = product(P, v)

Base.size(P::DistributedSum) = size(P.pm[1])

function position!(P::DistributedSum, psi::MPS, pos::Int)
  return @distributed for M in P.pm
    position!(fetch(M), psi, pos)
  end
end

function noiseterm(P::DistributedSum,
                   phi::ITensor,
                   dir::String)
  nts = fill(ITensor(), Threads.nthreads())
  Threads.@threads for M in P.pm
    nts[Threads.threadid()] += noiseterm(M, phi, dir)
  end
  return sum(nts)
end
