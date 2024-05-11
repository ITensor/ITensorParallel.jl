using Distributed

rmprocs(setdiff(procs(), 1))
addprocs(2)
@show nprocs()

@everywhere using ITensorMPS
@everywhere using ITensorParallel
@everywhere using ITensors
using LinearAlgebra: BLAS
using Random: Random
using Strided: Strided

electronk_path = joinpath(pkgdir(ITensors), "src", "lib", "ITensorMPS", "examples", "src")
include(joinpath(electronk_path, "electronk.jl"))
include(joinpath(electronk_path, "hubbard.jl"))

BLAS.set_num_threads(1)
Strided.disable_threads()

"""
Run with:
```julia
# Sequential sum over MPOs.
# Uses the default `Sum=SequentialSum`.
main(; Nx=8, Ny=4, nsweeps=10, maxdim=1000);
main(; Nx=8, Ny=4, nsweeps=10, maxdim=1000, threaded_blocksparse=true);

# Threaded sum over MPOs.
main(; Nx=8, Ny=4, nsweeps=10, maxdim=1000, Sum=ThreadedSum);
main(; Nx=8, Ny=4, nsweeps=10, maxdim=1000, Sum=ThreadedSum, threaded_blocksparse=true);

# Distributed sum over MPOs, where terms of the MPO
# sum and their environments are stored, updated,
# and applied remotely on a worker process.
main(; Nx=8, Ny=4, nsweeps=10, maxdim=1000, Sum=DistributedSum);
main(; Nx=8, Ny=4, nsweeps=10, maxdim=1000, Sum=DistributedSum, threaded_blocksparse=true);

# Using write-to-disk.
main(; Nx=8, Ny=4, maxdim=1000, Sum=DistributedSum, disk=true, threaded_blocksparse=true);
```
"""
function main(;
  Nx::Int,
  Ny::Int,
  U::Float64=4.0,
  t::Float64=1.0,
  nsweeps=10,
  maxdim::Int=3000,
  conserve_ky=true,
  seed=1234,
  npartitions=2Ny,
  Sum=SequentialSum,
  threaded_blocksparse=false,
  disk=false,
  random_init=false,
  in_partition_alg="sum_split",
)
  @show Threads.nthreads()

  ITensors.enable_threaded_blocksparse(threaded_blocksparse)
  @show ITensors.using_threaded_blocksparse()

  N = Nx * Ny

  max_maxdim = maxdim
  maxdim = min.([100, 200, 400, 800, 2000, 3000, max_maxdim], max_maxdim)
  cutoff = 1e-6
  noise = [1e-6, 1e-7, 1e-8, 0.0]
  @show nsweeps
  @show maxdim
  @show cutoff
  @show noise

  sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky=conserve_ky, modulus_ky=Ny)

  ℋ = hubbard(; Nx=Nx, Ny=Ny, t=t, U=U, ky=true)
  ℋs = partition(ℋ, npartitions; in_partition_alg)
  Hs = [MPO(ℋ, sites) for ℋ in ℋs]

  @show maxlinkdim.(Hs)

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  @show nnz(Hs[1][end ÷ 2])
  @show nnzblocks(Hs[1][end ÷ 2])

  # Create starting state with checkerboard
  # pattern
  state = map(CartesianIndices((Ny, Nx))) do I
    return iseven(I[1]) ⊻ iseven(I[2]) ? "↓" : "↑"
  end
  display(state)

  if random_init
    # Only available in ITensors 0.3.27
    # Helps make results reproducible when comparing
    # sequential vs. threaded.
    itensor_rng = Xoshiro()
    Random.seed!(itensor_rng, seed)
    psi0 = randomMPS(itensor_rng, sites, state; linkdims=10)
  else
    psi0 = MPS(sites, state)
  end

  mpo_sum = Sum(Hs)
  if disk
    # Write-to-disk
    mpo_sum = ITensors.disk(mpo_sum)
  end

  energy, psi = @time dmrg(mpo_sum, psi0; nsweeps, maxdim, cutoff, noise)
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  return energy, psi
end
