using MPI

MPI.Init()

comm = MPI.COMM_WORLD
N = 4
root = 0

if MPI.Comm_rank(comm) == root
  v = randn(N)
else
  v = Vector{Float64}(undef, N)
end

MPI.Bcast!(v, root, comm)

M = randn(N, N)

Mv = M * v

@show Mv

Mv_total = MPI.Reduce(Mv, +, root, comm)

@show MPI.Comm_rank(comm), Mv_total

MPI.Finalize()
