# ITensorParallel

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mtfishman.github.io/ITensorParallel.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mtfishman.github.io/ITensorParallel.jl/dev)
[![Build Status](https://github.com/mtfishman/ITensorParallel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mtfishman/ITensorParallel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mtfishman/ITensorParallel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mtfishman/ITensorParallel.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

# Overview

This package is for experimenting with adding more shared and distributed memory parallelism, for example implementing the techniques laid out in the paper https://arxiv.org/abs/2103.09976.

We will explore multithreaded and distributed parallelism over real-space parallel DMRG, TDVP, and TEBD based on https://arxiv.org/abs/1301.3494, as well as multithreaded and distributed parallelism over sums of Hamiltonian terms in DMRG and TDVP.

For multithreading, we are using Julia's standard Threads.jl library, and possibly convenient abstractions on top of that provided by [Folds.jl](https://github.com/juliafolds/folds.jl) and [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl). See [here](https://juliafolds.github.io/data-parallelism/tutorials/quick-introduction/) for a nice overview of the different options.

For distributed computing, we will explore Julia's standard [Distributed.jl](https://docs.julialang.org/en/v1/manual/distributed-computing/) library, as well as [MPI.jl](https://juliaparallel.github.io/MPI.jl/latest/).

To run Distributed.jl-based computations on clusters, we will explore using Julia's cluster manager tools like [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl) and [SlurmClusterManager.jl](https://github.com/kleinhenz/SlurmClusterManager.jl).

# Running on clusters

Here are detailed instructions for running a minimal "hello world" example parallelized with Julia's Distributed.jl standard library, distributed over nodes of a cluster.

1. Start by [downloading the latest version of Julia](https://julialang.org/downloads/) or loading a pre-installed version of Julia, for example with `module load julia`. You can follow more detailed instruction for installing your own version of Julia on a cluster [here](https://itensor.github.io/ITensors.jl/stable/getting_started/Installing.html).
2. Start Julia by executing the command `julia` at the command line. This should bring up the interactive Julia REPL. From there, you should install either [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl) or [SlurmClusterManager.jl](https://github.com/kleinhenz/SlurmClusterManager.jl) if your cluster uses Slurm as the cluster management and job scheduling system (ClusterManagers.jl supports Slurm, but SlurmClusterManager.jl has a more specialized implementation). This would look something like this:
```julia
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.7.2 (2022-02-06)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> using Pkg

julia> Pkg.add("SlurmClusterManager")
    Updating registry at `~/.julia/registries/General`
    Updating git-repo `https://github.com/JuliaRegistries/General.git`
   Resolving package versions...
   Installed SlurmClusterManager ─ v0.1.2
    Updating `~/.julia/environments/v1.7/Project.toml`
  [c82cd089] + SlurmClusterManager v0.1.2
    Updating `~/.julia/environments/v1.7/Manifest.toml`
  [c82cd089] + SlurmClusterManager v0.1.2
Precompiling project...
  1 dependency successfully precompiled in 2 seconds (456 already precompiled, 2 skipped during auto due to previous errors)
```
Now the cluster manager (either `ClusterManagers.jl` or `SlurmClusterManager.jl`, whichever you installed) will be available system-wide for you to use and aid you in running your Distributed.jl-based parallelized Julia code across nodes of your cluster.
3. Create a file somewhere within your home directory on the cluster called `hello_world.jl` with the contents:
```julia
#!/usr/bin/env julia

using Distributed, SlurmClusterManager
addprocs(SlurmManager())
@everywhere println("hello from $(myid()):$(gethostname())")
```
4. Submit your script to the work queue of your cluster, for example with `sbatch` if the cluster you are on uses Slurm:
```
$ sbatch -N 2 --ntasks-per-node=64 hello_world.jl
```
This will execute the code on two nodes using 64 workers per node.

We will add similar instructions on running the same "hello world" example using MPI.jl, and additionally running linear algebra and ITensor operations in parallel with both Distributed.jl and MPI.jl.
