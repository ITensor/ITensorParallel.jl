struct MPISumTerm{T}
  term::T
  comm::MPI.Comm
end
term(sumterm::MPISumTerm) = sumterm.term
comm(sumterm::MPISumTerm) = sumterm.comm

set_term(sumterm::MPISumTerm, term) = (@set sumterm.term = term)
set_comm(sumterm::MPISumTerm, comm) = (@set sumterm.comm = comm)

MPISumTerm(mpo::MPO, comm::MPI.Comm) = MPISumTerm(ProjMPO(mpo), comm)

nsite(sumterm::MPISumTerm) = nsite(term(sumterm))

length(sumterm::MPISumTerm) = length(term(sumterm))

disk(sumterm::MPISumTerm) = set_term(sumterm, disk(term(sumterm)))

function product(sumterm::MPISumTerm, v::ITensor)
  return allreduce(term(sumterm)(v), +, comm(sumterm))
end

function eltype(sumterm::MPISumTerm)
  return eltype(term(sumterm))
end

(sumterm::MPISumTerm)(v::ITensor) = product(sumterm, v)

size(sumterm::MPISumTerm) = size(term(sumterm))

# Replace the bond and then broadcast one of the results
# to each process. This ensures that the link indices
# are consistent across processes and that the tensors
# match numerically. Floating point precision errors
# have been seen to build up enough to make their
# block sparse structure inconsistent, especially
# when multithreading block sparse operations are involved.
function replacebond!(sumterm::MPISumTerm, M::MPS, b::Int, v::ITensor; kwargs...)
  spec = nothing
  M_ortho_lims = nothing
  if MPI.Comm_rank(comm(sumterm)) == 0
    spec = replacebond!(M, b, v; kwargs...)
    M_ortho_lims = ortho_lims(M)
  end
  M_b1 = MPI.bcast(M[b], 0, comm(sumterm))
  M_b2 = MPI.bcast(M[b + 1], 0, comm(sumterm))
  M_ortho_lims = MPI.bcast(M_ortho_lims, 0, comm(sumterm))
  spec = MPI.bcast(spec, 0, comm(sumterm))
  M[b] = M_b1
  M[b + 1] = M_b2
  set_ortho_lims!(M, M_ortho_lims)
  return spec
end

function orthogonalize!(sumterm::MPISumTerm, v::MPS, pos::Int; kwargs...)
  if MPI.Comm_rank(comm(sumterm)) == 0
    v = orthogonalize!(v, pos; kwargs...)
  end
  return MPI.bcast(v, 0, comm(sumterm))
end

function position!(sumterm::MPISumTerm, v::MPS, pos::Int)
  return set_term(sumterm, position!(term(sumterm), v, pos))
end

function noiseterm(sumterm::MPISumTerm, v::ITensor, dir::String)
  # TODO: Check if this logic is correct.
  # Should the noiseterm instead be calculated on the reduced `sumterm(v)` and then broadcasted?
  return allreduce(noiseterm(term(sumterm), v, dir), +, comm(sumterm))
end
