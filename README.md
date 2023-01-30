# ITensorParallel

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mtfishman.github.io/ITensorParallel.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mtfishman.github.io/ITensorParallel.jl/dev)
[![Build Status](https://github.com/mtfishman/ITensorParallel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mtfishman/ITensorParallel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mtfishman/ITensorParallel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mtfishman/ITensorParallel.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

# Overview

This package adds more shared and distributed memory parallelism to [ITensors.jl](https://github.com/ITensor/ITensors.jl), for example implementing the techniques for nested parallelization in DMRG laid out in the paper [arXiv:2103.09976](https://arxiv.org/abs/2103.09976). So far, it focuses on parallelizing over optimizing or evolving sums of tensor networks. We plan to add real-space parallel DMRG, TDVP, and TEBD based on [arXiv:1301.3494](https://arxiv.org/abs/1301.3494).

For multithreading, we are using Julia's standard Threads.jl library, as well as convenient abstractions on top of that for parallelizing over maps and reductions provided by [Folds.jl](https://github.com/juliafolds/folds.jl) and [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl). See [here](https://juliafolds.github.io/data-parallelism/tutorials/quick-introduction/) for a nice overview of parallelization in Julia.

For distributed computing, we make use of Julia's standard [Distributed.jl](https://docs.julialang.org/en/v1/manual/distributed-computing/) library along with it's interface through [Folds.jl](https://github.com/juliafolds/folds.jl) and [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl), as well as [MPI.jl](https://juliaparallel.github.io/MPI.jl/latest/). Take a look at Julia'd documentation on [distributed computing](https://docs.julialang.org/en/v1/manual/distributed-computing/) for more information and background.

To run Distributed.jl-based computations on clusters, we recommend using Julia's cluster manager tools like [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl), [SlurmClusterManager.jl](https://github.com/kleinhenz/SlurmClusterManager.jl), and [MPIClusterManagers.jl](https://github.com/JuliaParallel/MPIClusterManagers.jl).

See the [examples folder](https://github.com/ITensor/ITensorParallel.jl/tree/main/examples) for examples of running DMRG parallelized over sums of Hamiltonians, using Threads.jl, Distributed.jl, and MPI.jl.
