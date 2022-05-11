struct MPISum{T}
  data::T
  comm::MPI.Comm
end

MPISum(data::T, comm=MPI.COMM_WORLD) where {T} = MPISum{T}(data, comm)

# Apply the sum
function (A::MPISum)(v)
  return MPI.Allreduce(A.data(v), +, A.comm)
end
