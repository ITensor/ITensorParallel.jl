function MPI.RBuffer(senddata::ITensor, recvdata::ITensor)
  senddata = permute(senddata, inds(recvdata); allow_alias=true)
  @assert inds(senddata) == inds(recvdata)
  _senddata = ITensors.data(senddata)
  _recvdata = ITensors.data(recvdata)
  return MPI.RBuffer(_senddata, _recvdata)
end

struct MPISum{T}
  data::T
  comm::MPI.Comm
end

MPISum(data::T, comm=MPI.COMM_WORLD) where {T} = MPISum{T}(data, comm)

nsite(P::MPISum) = nsite(P.data)

Base.length(P::MPISum) = length(P.data)

function _Allreduce(sendbuf::ITensor, op, comm)
  return _Allreduce(NDTensors.storagetype(tensor(sendbuf)), sendbuf, op, comm)
end

function _Allreduce(::Type{<:BlockSparse}, sendbuf::ITensor, op, comm)
  # Similar to:
  #
  # recvbuf = ITensor(eltype(sendbuf), flux(sendbuf), inds(sendbuf))
  #
  # But more efficient.
  nprocs = MPI.Comm_size(comm)
  N = order(sendbuf)
  max_nblocks = maximum(MPI.Allgather(nnzblocks(sendbuf), comm))
  blocks_sendbuf = fill(0, max_nblocks, N)
  nzblocks_matrix = Matrix{Int}(reduce(hcat, collect.(nzblocks(sendbuf)))')
  blocks_sendbuf[1:size(nzblocks_matrix, 1), :] = nzblocks_matrix
  allblocks = MPI.Allgather(blocks_sendbuf, comm)
  allblocks_tensor = reshape(allblocks, max_nblocks, N, nprocs)
  allblocks_vector = [allblocks_tensor[:, :, i] for i in 1:nprocs]
  allblocks_vector_tuple = [
    [Tuple(allblocks_vector[n][i, :]) for i in 1:max_nblocks] for n in 1:nprocs
  ]
  blocks = Block.(filter(≠(ntuple(Returns(0), N)), union(allblocks_vector_tuple...)))
  recvbuf = itensor(BlockSparseTensor(eltype(sendbuf), blocks, inds(sendbuf)))
  sendbuf_filled = copy(recvbuf)
  sendbuf_filled .= sendbuf
  MPI.Allreduce!(sendbuf_filled, recvbuf, +, comm)
  return recvbuf
end

function _Allreduce(::Type{<:Dense}, sendbuf::ITensor, op, comm)
  recvbuf = similar(sendbuf)
  MPI.Allreduce!(sendbuf, recvbuf, op, comm)
  return recvbuf
end

function product(P::MPISum, v::ITensor)
  return _Allreduce(P.data(v), +, P.comm)
end

function Base.eltype(P::MPISum)
  return eltype(P.data)
end

(P::MPISum)(v::ITensor) = product(P, v)

Base.size(P::MPISum) = size(P.data)

function position!(P::MPISum, psi::MPS, pos::Int)
  position!(P.data, psi, pos)
  return P
end

## XXX: Implement this.
## function noiseterm(P::MPISum,
##                    phi::ITensor,
##                    dir::String)
##   return noiseterm(P.data, phi, dir)
## end
