
"""
bcast(obj, comm::Comm; root::Integer=Cint(0)) =
    bcast(obj, root, comm)
function bcast(obj, root::Integer, comm::Comm)
    isroot = Comm_rank(comm) == root
    count = Ref{Cint}()
    if isroot
        buf = MPI.serialize(obj)
        count[] = length(buf)
    end
    Bcast!(count, root, comm)
    if !isroot
        buf = Array{UInt8}(undef, count[])
    end
    Bcast!(buf, root, comm)
    if !isroot
        obj = MPI.deserialize(buf)
    end
    return obj
end
"""

_gather(obj, comm::MPI.Comm, root::Integer=Cint(0)) = _gather(obj, root, comm)
function _gather(obj, root::Integer, comm::MPI.Comm)
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
