using Distributed

# Number of matrices in the sum
nmats = 2
addprocs(nmats)

@everywhere begin
  using LinearAlgebra
  using KrylovKit

  BLAS.set_num_threads(1)

  include(joinpath("src", "DistributedSums.jl"))
  include(joinpath("src", "MatrixMaps.jl"))

  # Matrix dimension
  n = 100

  # Starting vector that we will apply
  # the matrix map to
  v0 = randn(n)
end

# Make a distributed sum of MatrixMaps, which is
# just a Matrix with special multiplication
# notation that can work with KrylovKit
A = DistributedSum(_ -> MatrixMap(randn(n, n)), nmats)

Av0 = A(v0)

@show norm(Av0 - sum(fetch.(A))(v0))

# Solve eigenvector equation
λ⃗, v⃗ = eigsolve(A, v0, 1, :LM)
λ = λ⃗[1]
v = v⃗[1]

@show norm(A(v) - λ * v)
