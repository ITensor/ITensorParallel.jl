using ITensors
using ITensorParallel
using Random

include(joinpath(pkgdir(ITensors), "examples", "src", "electronk.jl"))
include(joinpath(pkgdir(ITensors), "examples", "src", "hubbard.jl"))

function main(;
  Nx::Int=6,
  Ny::Int=3,
  U::Float64=4.0,
  t::Float64=1.0,
  maxdim::Int=3000,
  conserve_ky=true,
  use_splitblocks=true,
  seed=1234,
  in_partition=ITensorParallel.default_in_partition,
)
  Random.seed!(seed)
  @show Threads.nthreads()
  @show ITensors.using_threaded_blocksparse()

  N = Nx * Ny

  nsweeps = 10
  maxdims = min.([100, 200, 400, 800, 2000, 3000, maxdim], maxdim)
  cutoff = 1e-6
  noise = [1e-6, 1e-7, 1e-8, 0.0]
  @show nsweeps
  @show maxdims
  @show cutoff
  @show noise

  sites = siteinds("ElecK", N; conserve_qns=true, conserve_ky=conserve_ky, modulus_ky=Ny)

  ℋ = hubbard(; Nx=Nx, Ny=Ny, t=t, U=U, ky=true)
  ℋs = partition(ℋ, Threads.nthreads(); in_partition)
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

  energy, psi = @time dmrg(
    ThreadedSum(H), psi0; nsweeps, maxdims, cutoff, noise
  )
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
  return energy, H, psi
end

function custom_in_partition(sites::Tuple, p, nparts)
  return p == mod1(sites[1], nparts)
end

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()
ITensors.disable_threaded_blocksparse()

# A function that specifies which partition/processor that an
# MPO term will be on.
in_partition = ITensorParallel.default_in_partition # or: custom_in_partition
main(; in_partition)
