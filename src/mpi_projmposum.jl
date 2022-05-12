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

function product(P::MPISum, v::ITensor)::ITensor
  # Error: Type must be isbitstype
  # return MPI.Allreduce(P.data(v), +, P.comm)
  Pv = similar(v)
  MPI.Allreduce!(P.data(v), Pv, +, P.comm)
  return Pv
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
