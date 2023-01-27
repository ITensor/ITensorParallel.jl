using Distributed

rmprocs(setdiff(procs(), 1))
addprocs(4)
@show nprocs()

@everywhere using ITensors
@everywhere using ITensorParallel
using Random

include(joinpath(pkgdir(ITensors), "examples", "src", "electronk.jl"))
include(joinpath(pkgdir(ITensors), "examples", "src", "hubbard.jl"))

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

"""
Run with:
```julia
# No blocksparse multithreading
main(; Nx=8, Ny=4, maxdim=1000, Sum=ThreadedSum);
main(; Nx=8, Ny=4, maxdim=1000, Sum=DistributedSum);
main(; Nx=8, Ny=4, maxdim=1000, Sum=SequentialSum);

# Blocksparse multithreading
main(; Nx=8, Ny=4, maxdim=1000, Sum=ThreadedSum, threaded_blocksparse=true);
main(; Nx=8, Ny=4, maxdim=1000, Sum=DistributedSum, threaded_blocksparse=true);
main(; Nx=8, Ny=4, maxdim=1000, Sum=SequentialSum, threaded_blocksparse=true);
```
"""
function main(;
  Nx::Int,
  Ny::Int,
  U::Float64=4.0,
  t::Float64=1.0,
  maxdim::Int=3000,
  conserve_ky=true,
  seed=1234,
  npartitions=2Ny,
  Sum,
  threaded_blocksparse=false,
  in_partition=ITensorParallel.default_in_partition,
)
  Random.seed!(seed)
  @show Threads.nthreads()

  if threaded_blocksparse
    ITensors.enable_threaded_blocksparse()
  else
    ITensors.disable_threaded_blocksparse()
  end
  @show ITensors.using_threaded_blocksparse()

  N = Nx * Ny

  nsweeps = 10
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
  ℋs = partition(ℋ, npartitions; in_partition)
  H = [MPO(ℋ, sites) for ℋ in ℋs]

  @show maxlinkdim.(H)

  # Number of structural nonzero elements in a bulk
  # Hamiltonian MPO tensor
  @show nnz(H[1][end ÷ 2])
  @show nnzblocks(H[1][end ÷ 2])

  # Create start state
  state = Vector{String}(undef, N)
  for i in 1:N
    x = (i - 1) ÷ Ny
    y = (i - 1) % Ny
    if x % 2 == 0
      if y % 2 == 0
        state[i] = "Up"
      else
        state[i] = "Dn"
      end
    else
      if y % 2 == 0
        state[i] = "Dn"
      else
        state[i] = "Up"
      end
    end
  end

  psi0 = randomMPS(sites, state; linkdims=10)

  energy, psi = @time dmrg(Sum(H), psi0; nsweeps, maxdim, cutoff, noise)
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  return energy, H, psi
end
