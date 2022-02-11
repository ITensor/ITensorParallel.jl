using Distributed

# Number of ITensorNetworkMaps in the sum
nmats = 2

addprocs(nmats)

# Set up for the calculation. Define everything
# in the `@everywhere` block that you need
# on all of the processes to perform the
# calculation.
@everywhere begin
  using ITensors
  using ITensors.ITensorNetworkMaps
  using LinearAlgebra
  using KrylovKit

  BLAS.set_num_threads(1)
  ITensors.Strided.disable_threads()
  ITensors.disable_threaded_blocksparse()

  include(joinpath("src", "DistributedSums.jl"))

  # Function that defines the tensor network
  # that will map the input vector (i.e.
  # a transfer matrix or effective Hamiltonian).
  function itensor_map(i, j)
    A1 = randomITensor(i', dag(i), dag(l))
    A2 = randomITensor(j', dag(j), l)
    return ITensorNetworkMap([A1, A2])
  end

  d = 2
  χ = 3
  i = Index(d, "i")
  j = Index(d, "j")
  l = Index(χ, "l")

  v0 = randomITensor(i, j)
end

# Defines an object representing a sum of
# linear maps (generalized matrices) built
# from contractions of ITensors that are
# distributed across `nmats` workers
# (each linear map lives on a worker).
A = DistributedSum(_ -> itensor_map(i, j), nmats)

# Apply the linear map, which internally
# is performed in parallel where the linear
# map is created.
Av0 = A(v0)

@show norm(Av0 - sum(fetch.(A))(v0))

# Solve an eigenvector equation in
# parallel.
λ⃗, v⃗ = eigsolve(A, v0, 1, :LM)
λ = λ⃗[1]
v = v⃗[1]

@show norm(A(v) - λ * v)
