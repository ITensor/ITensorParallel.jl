using ITensors
using ITensorParallel
using MPI
using Random

MPI.Init()

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()
ITensors.disable_threaded_blocksparse()

include("electronk.jl")
include("hubbard_ky.jl")

U = 10.0
Nx = 4
Ny = 2
N = Nx * Ny

@show U, Nx, Ny
sites = siteinds("ElectronK", N; conserve_qns=true, conserve_ky=true, modulus_ky=Ny)

opsums = hubbard_ky(; Nx, Ny, U)

# TODO: randomly shuffle the MPOs
# TODO: only construct the MPOs that are needed by the current process
Hs = [splitblocks(linkinds, MPO(opsum, sites)) for opsum in opsums]

nprocs = MPI.Comm_size(MPI.COMM_WORLD)
Hss = collect(Iterators.partition(Hs, nprocs))

n = MPI.Comm_rank(MPI.COMM_WORLD) + 1
Hn = copy(Hss[n])

ITensors.checkflux.(Hn)
@show flux.(Hn)

PH = MPISum(ProjMPOSum(Hn))

init_state = Vector{String}(undef, N)
for i in 1:N
  x = (i - 1) ÷ Ny
  y = (i - 1) % Ny
  if x % 2 == 0
    if y % 2 == 0
      init_state[i] = "Up"
    else
      init_state[i] = "Dn"
    end
  else
    if y % 2 == 0
      init_state[i] = "Dn"
    else
      init_state[i] = "Up"
    end
  end
end

Random.seed!(1234)

psi0 = randomMPS(sites, init_state; linkdims=20)
ITensors.checkflux(psi0)
@show flux(psi0)
dmrg_kwargs = (nsweeps=20, maxdim=[10, 20, 50, 100, 200], cutoff=1e-5, noise=1e-4)
energy, psi = @time dmrg(PH, psi0; dmrg_kwargs...)

MPI.Finalize()
