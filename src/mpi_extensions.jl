gather(obj, comm::MPI.Comm, root::Integer=Cint(0)) = gather(obj, root, comm)
function gather(obj, root::Integer, comm::MPI.Comm)
  isroot = MPI.Comm_rank(comm) == root
  count = Ref{Cint}()
  buf = MPI.serialize(obj)
  count[] = length(buf)
  counts = MPI.Gather(count[], root, comm)
  if isroot
    rbuf = Array{UInt8}(undef, reduce(+, counts))
    rbuf = MPI.VBuffer(rbuf, counts)
  else
    rbuf = nothing
  end
  MPI.Gatherv!(buf, rbuf, root, comm)
  if isroot
    objs = []
    for v in 1:length(rbuf.counts)
      endind = v == length(rbuf.counts) ? length(rbuf.data) : rbuf.displs[v + 1] + 1
      startind = rbuf.displs[v] + 1
      push!(objs, MPI.deserialize(rbuf.data[startind:endind]))
    end
  else
    objs = nothing
  end
  return objs
end

function allreduce(sendbuf, op, comm::MPI.Comm)
  ##maybe better to implement as allgather with local reduce, but higher communication cost associated
  bufs = gather(sendbuf, 0, comm)
  rank = MPI.Comm_rank(comm)
  if rank == 0
    res = reduce(op, bufs)
  else
    res = nothing
  end
  return MPI.bcast(res, 0, comm)
end
