using MPI
using ITensors
using ITensorParallel
using Random

include(joinpath(pkgdir(ITensors), "examples", "src", "electronk.jl"))
include(joinpath(pkgdir(ITensors), "examples", "src", "hubbard.jl"))

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

# Run with `main()`
function main(;
  Nx::Int=6,
  Ny::Int=3,
  U::Float64=4.0,
  t::Float64=1.0,
  maxdim::Int=3000,
  conserve_ky=true,
  seed=1234,
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

  MPI.Init()

  ℋs = partition(ℋ, MPI.Comm_size(MPI.COMM_WORLD); in_partition)
  n = MPI.Comm_rank(MPI.COMM_WORLD) + 1
  PH = MPISum(ProjMPO(MPO(ℋs[n], sites)))
  energy, psi = @time dmrg(
    PH, psi0; nsweeps, maxdim, cutoff, noise
  )

  MPI.Finalize()

  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  return energy, H, psi
end

main()
