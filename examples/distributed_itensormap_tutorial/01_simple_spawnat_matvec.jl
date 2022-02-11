using Distributed

# Apply a sum of matrices to a vector in
# parallel using multi-processing.
# The vector exists on every process, but
# each matrix in the sum only exists on
# a single process.

# Set the number of processors Julia will use
# @everywhere denotes that it will be defined
# on every worker.
@everywhere nprocs = 2
addprocs(nprocs)

# Define the vector dimension.
# @everywhere denotes that it will be defined
# on every worker.
@everywhere n = 4

# Define the vector we will apply the set
# of matrices to.
# @everywhere denotes that it will be defined
# on every worker.
@everywhere v = randn(n)

# Define the matrices in the matrix sum
# that will be applied in parallel to the
# vector `v`.
# The first matrix is defined on worker 1,
# the second one on worker 2.
A1 = @spawnat workers()[1] randn(n, n)
A2 = @spawnat workers()[2] randn(n, n)

# Define the individual matrix-vector multiplication
# operations on the appropriate worker.
# Alternatively, you can use:
#
# A1v = @spawnat :any fetch(A1) * v
#
# for Julia to choose the worker automatically.
A1v = @spawnat workers()[1] fetch(A1) * v
A2v = @spawnat workers()[2] fetch(A2) * v

# Actually perform the matrix-vector multiplications,
# fetching the results to the main worker for
# local usage.
Av = fetch(A1v) + fetch(A2v)

@show norm((fetch(A1) + fetch(A2)) * v - Av)
