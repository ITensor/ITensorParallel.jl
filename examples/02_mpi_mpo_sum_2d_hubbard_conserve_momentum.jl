using MPI
using ITensors
using ITensorParallel
using Random

include(joinpath(pkgdir(ITensors), "examples", "src", "electronk.jl"))
include(joinpath(pkgdir(ITensors), "examples", "src", "hubbard.jl"))

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

"""
Run at the command line with 4 processes:
```julia
mpiexecjl -n 2 julia 02_mpi_run.jl --Nx 8 --Ny 4 --maxdim 20

# Currently is broken!
mpiexecjl -n 2 julia -t2 02_mpi_run.jl --Nx 8 --Ny 4 --maxdim 20 --threaded_blocksparse true
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
  threaded_blocksparse=false,
  in_partition=ITensorParallel.default_in_partition,
)
  # It is very important to set both of these RNG seeds!
  # This ensures the same state and the same ITensor indices
  # are made in each process
  Random.seed!(seed)
  Random.seed!(index_id_rng(), seed)

  @show Threads.nthreads()

  # TODO: Use `ITensors.enable_threaded_blocksparse(threaded_blocksparse)`
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

  # Create a lazy representation of the Hamiltonian
  ℋ = hubbard(; Nx, Ny, t, U, ky=true)

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
  sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky, modulus_ky=Ny)
  psi0 = randomMPS(sites, state; linkdims=10)

  MPI.Init()
  nprocs = MPI.Comm_size(MPI.COMM_WORLD)
  ℋs = partition(ℋ, nprocs; in_partition)
  which_proc = MPI.Comm_rank(MPI.COMM_WORLD) + 1
  PH = MPISum(ProjMPO(MPO(ℋs[which_proc], sites)))
  energy, psi = @time dmrg(PH, psi0; nsweeps, maxdim, cutoff, noise)

  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  return energy, psi
end
