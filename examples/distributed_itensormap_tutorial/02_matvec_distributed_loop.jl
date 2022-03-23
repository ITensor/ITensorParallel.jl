using Distributed
using LinearAlgebra

BLAS.set_num_threads(1)

# Apply a sum of distributed matrices and reduce
# after applying to a vector.

# Number of matrices in our sum.
nmats = 2

# Specify the number of processors Julia will use.
# For now we just say it is the same as the number of
# matrices in our sum.
addprocs(nmats)

# Size of out matrices.
@everywhere n = 4

# Set of matrices in our sum, each created on a
# separate worker.
As = [@spawnat(workers()[nmat], randn(n, n)) for nmat in 1:nmats]

# Vector we will apply to our matrices, distributed to
# every worker.
@everywhere v = randn(n)

# Distributed matrix-vector multiplication for each distributed
# matrix in our sum, then reduced over with `+`.
Av = @distributed (+) for n in 1:nmats
  # Need to call `fetch` to instantiate the matrix
  # on the worker where it was defined above.
  fetch(As[n]) * v
end

# Test by fetching and summing locally
A = sum(fetch.(As))
@show norm(Av - A * v)
