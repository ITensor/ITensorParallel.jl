# ITensorParallel

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mtfishman.github.io/ITensorParallel.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mtfishman.github.io/ITensorParallel.jl/dev)
[![Build Status](https://github.com/mtfishman/ITensorParallel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mtfishman/ITensorParallel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mtfishman/ITensorParallel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mtfishman/ITensorParallel.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

This package is for experimenting with adding more shared and distributed memory parallelism, for example implementing the techniques laid out in the paper https://arxiv.org/abs/2103.09976.

We will explore multithreaded and distributed parallelism over real-space parallel DMRG, TDVP, and TEBD based on https://arxiv.org/abs/1301.3494, as well as multithreaded and distributed parallelism over sums of Hamiltonian terms in DMRG and TDVP.

For multithreading, we are using Julia's standard Threads.jl library, and possibly convenient abstractions on top of that provided by [Folds.jl](https://github.com/juliafolds/folds.jl) and [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl). See [here](https://juliafolds.github.io/data-parallelism/tutorials/quick-introduction/) for a nice overview of the different options.

For distributed computing, we will explore Julia's standard [Distributed.jl](https://docs.julialang.org/en/v1/manual/distributed-computing/) library, as well as [MPI.jl](https://juliaparallel.github.io/MPI.jl/latest/).

To run Distributed.jl-based computations on clusters, we will explore using Julia's cluster manager tools like [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl) and [SlurmClusterManager.jl](https://github.com/kleinhenz/SlurmClusterManager.jl).
