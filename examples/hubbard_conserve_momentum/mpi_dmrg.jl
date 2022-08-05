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
include("split_vec.jl")

U = 10.0
Nx = 2
Ny = 2
N = Nx * Ny

which_proc = MPI.Comm_rank(MPI.COMM_WORLD) + 1

@show which_proc, U, Nx, Ny
sites = siteinds("ElectronK", N; conserve_qns=true, conserve_ky=true, modulus_ky=Ny)

opsums = hubbard_ky(; Nx, Ny, U)

@show which_proc, length.(opsums)

# TODO: randomly shuffle the MPOs
# TODO: only construct the MPOs that are needed by the current process
Hs = [splitblocks(linkinds, MPO(opsum, sites)) for opsum in opsums]

@show which_proc, length(Hs)

nprocs = MPI.Comm_size(MPI.COMM_WORLD)

@show which_proc, nprocs

Hss = split_vec(Hs, nprocs)

@show which_proc, length.(Hss)
@show which_proc, [maxlinkdim.(Hs) for Hs in Hss]

Hn = Hss[which_proc]

@show which_proc, length(Hss[which_proc])

ITensors.checkflux.(Hss[which_proc])
@show flux.(Hss[which_proc])

PH = MPISum(ProjMPOSum(Hss[which_proc]))

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
dmrg_kwargs = (nsweeps=10, maxdim=[10, 20, 50, 100], cutoff=1e-5, noise=1e-4)
energy, psi = @time dmrg(PH, psi0; dmrg_kwargs...)

MPI.Finalize()
