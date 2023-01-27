struct MPISum{T}
  term::T
  comm::MPI.Comm
end
term(sum::MPISum) = sum.term
comm(sum::MPISum) = sum.comm

MPISum(term::T, comm=MPI.COMM_WORLD) where {T} = MPISum{T}(term, comm)

MPISum(mpo::MPO, comm=MPI.COMM_WORLD) = MPISum(ProjMPO(mpo), comm)

nsite(sum::MPISum) = nsite(term(sum))

length(sum::MPISum) = length(term(sum))

function product(sum::MPISum, v::ITensor)
  return allreduce(term(sum)(v), +, comm(sum))
end

function eltype(sum::MPISum)
  return eltype(term(sum))
end

(sum::MPISum)(v::ITensor) = product(sum, v)

size(sum::MPISum) = size(term(sum))

function position!(sum::MPISum{T}, psi::MPS, pos::Int) where {T<:ITensors.AbstractProjMPO}
  makeL!(sum, psi, pos - 1)
  makeR!(sum, psi, pos + nsite(sum))
  return sum
end

function makeL!(sum::MPISum, psi::MPS, k::Int)
  _makeL!(sum, psi, k)
  return sum
end

function makeR!(sum::MPISum, psi::MPS, k::Int)
  _makeR!(sum, psi, k)
  return sum
end

function _makeL!(sum::MPISum, psi::MPS, k::Int)::Union{ITensor,Nothing}
  # Save the last `L` that is made to help with caching
  # for DiskProjMPO
  sum_term = term(sum)
  ll = sum_term.lpos
  if ll ≥ k
    # Special case when nothing has to be done.
    # Still need to change the position if lproj is
    # being moved backward.
    sum_term.lpos = k
    return nothing
  end
  # Make sure ll is at least 0 for the generic logic below
  ll = max(ll, 0)
  L = lproj(sum_term)
  while ll < k
    # sync the linkindex across processes
    mylink = linkind(psi, ll + 1)
    otherlink = MPI.bcast(mylink, 0, comm(sum))
    replaceind!(psi[ll + 1], mylink, otherlink)
    replaceind!(psi[ll + 2], mylink, otherlink)

    L = L * psi[ll + 1] * sum_term.H[ll + 1] * dag(prime(psi[ll + 1]))
    sum_term.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  sum_term.lpos = k
  return L
end

function _makeR!(sum::MPISum{ProjMPO}, psi::MPS, k::Int)::Union{ITensor,Nothing}
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
  sum_term = term(sum)
  rl = sum_term.rpos
  if rl ≤ k
    # Special case when nothing has to be done.
    # Still need to change the position if rproj is
    # being moved backward.
    sum_term.rpos = k
    return nothing
  end
  N = length(sum_term.H)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(sum_term)
  while rl > k
    #sync linkindex across processes
    mylink = linkind(psi, rl - 2)
    otherlink = MPI.bcast(mylink, 0, comm(sum))
    replaceind!(psi[rl - 2], mylink, otherlink)
    replaceind!(psi[rl - 1], mylink, otherlink)

    R = R * psi[rl - 1] * sum_term.H[rl - 1] * dag(prime(psi[rl - 1]))
    sum_term.LR[rl - 1] = R
    rl -= 1
  end
  sum_term.rpos = k
  return R
end

function noiseterm(sum::MPISum, phi::ITensor, dir::String)
  ##ToDo: I think the logic here is wrong.
  ##The noiseterm should be calculated on the reduced P*v and then broadcasted?
  return allreduce(noiseterm(term(sum), phi, dir), +, comm(sum))
end
