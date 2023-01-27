# WARNING: This is currently broken!
struct DistributedSum{T}
  terms::Vector{T}
end
terms(sum::DistributedSum) = sum.terms

function DistributedSum(f::Function, n::Integer)
  # Assign the terms of the sum to the first `n` workers.
  return DistributedSum([@spawnat(workers()[ni], f(ni)) for ni in 1:n])
end

nsite(sum::DistributedSum) = nsite(fetch(terms(sum)[1]))

Base.length(sum::DistributedSum) = length(fetch(terms(sum)[1]))

Base.getindex(sum::DistributedSum, n::Int) = getindex(terms(sum), n)
Base.setindex!(sum::DistributedSum, x, n::Int) = setindex!(terms(sum), x, n)

function product(sum::DistributedSum, v::ITensor)::ITensor
  # TODO: Use `Folds.sum`:
  # Folds.sum(term -> fetch(term)(v), terms(sum), DistributedEx())
  return @distributed (+) for M in terms(sum)
    fetch(M)(v)
  end
end

function Base.eltype(sum::DistributedSum)
  elT = eltype(terms(sum)[1])
  for n in 2:length(terms(sum))
    elT = promote_type(elT, eltype(terms(sum)[n]))
  end
  return elT
end

(sum::DistributedSum)(v::ITensor) = product(sum, v)

Base.size(sum::DistributedSum) = size(terms(sum)[1])

function position!(sum::DistributedSum, psi::MPS, pos::Int)
  # TODO: Try `Folds.map`, i.e.:
  # new_terms = Folds.map(terms(sum), DistributedEx()) do term
  #   return @spawnat(worker(term), position!(fetch(term), psi, pos))
  # end
  # return set_terms(sum, new_terms)
  @distributed for n in 1:length(terms(sum))
    M_fetched = fetch(sum[n])
    M_fetched = position!(M_fetched, psi, pos)
    sum[n] = @spawnat(workers()[n], M_fetched)
  end
  return sum
end

## XXX: Implement this.
## function noiseterm(sum::DistributedSum,
##                    phi::ITensor,
##                    dir::String)
##   nts = fill(ITensor(), Threads.nthreads())
##   Threads.@threads for M in terms(sum)
##     nts[Threads.threadid()] += noiseterm(M, phi, dir)
##   end
##   return sum(nts)
## end
