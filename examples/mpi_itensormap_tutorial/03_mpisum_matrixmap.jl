using MPI

MPI.Init()

include(joinpath("src", "MPISum.jl"))
include(joinpath("src", "MatrixMaps.jl"))

n = 4
v = ones(n)
A = MPISum(MatrixMap(randn(n, n)))
Av = A(v)

@show MPI.Comm_rank(A.comm), Av

MPI.Finalize()
