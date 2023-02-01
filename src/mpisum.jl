struct MPISum{T}
  term::T
  comm::MPI.Comm
end
term(sum::MPISum) = sum.term
comm(sum::MPISum) = sum.comm

set_term(sum::MPISum, term) = (@set sum.term = term)
set_comm(sum::MPISum, comm) = (@set sum.comm = comm)

MPISum(mpo::MPO, comm::MPI.Comm) = MPISum(ProjMPO(mpo), comm)

nsite(sum::MPISum) = nsite(term(sum))

length(sum::MPISum) = length(term(sum))

disk(sum::MPISum) = set_term(sum, disk(term(sum)))

function product(sum::MPISum, v::ITensor)
  return allreduce(term(sum)(v), +, comm(sum))
end

function eltype(sum::MPISum)
  return eltype(term(sum))
end

(sum::MPISum)(v::ITensor) = product(sum, v)

size(sum::MPISum) = size(term(sum))

# Replace the bond and then broadcast one of the results
# to each process. This ensures that the link indices
# are consistent across processes and that the tensors
# match numerically. Floating point precision errors
# have been seen to build up enough to make their
# block sparse structure inconsistent, especially
# when multithreading block sparse operations are involved.
function ITensors.replacebond!(sum::MPISum, M::MPS, b::Int, v::ITensor; kwargs...)
  spec = replacebond!(M, b, v; kwargs...)
  M_ortho_lims = ortho_lims(M)
  M_b1 = MPI.bcast(M[b], 0, comm(sum))
  M_b2 = MPI.bcast(M[b + 1], 0, comm(sum))
  M[b] = M_b1
  M[b + 1] = M_b2
  set_ortho_lims!(M, M_ortho_lims)
  return spec
end

function position!(sum::MPISum, v::MPS, pos::Int)
  return set_term(sum, position!(term(sum), v, pos))
end

function noiseterm(sum::MPISum, v::ITensor, dir::String)
  # TODO: Check if this logic is correct.
  # Should the noiseterm instead be calculated on the reduced `sum(v)` and then broadcasted?
  return allreduce(noiseterm(term(sum), v, dir), +, comm(sum))
end
