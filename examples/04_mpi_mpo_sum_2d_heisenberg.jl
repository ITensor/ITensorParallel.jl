using MPI
using ITensors
using ITensorParallel
using Random

MPI.Init()

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()
ITensors.disable_threaded_blocksparse()

seed = 1234
Random.seed!(seed)

mpi_projmpo = true

Nx, Ny = 4, 1
N = Nx * Ny

sites = siteinds("S=1/2", N; conserve_qns=true)

lattice = square_lattice(Nx, Ny; yperiodic=false)

function heisenberg_hamiltonian(lattice)
  ℋ = OpSum()
  for b in lattice
    ℋ += 0.5, "S+", b.s1, "S-", b.s2
    ℋ += 0.5, "S-", b.s1, "S+", b.s2
    ℋ += "Sz", b.s1, "Sz", b.s2
  end
  return ℋ
end
ℋ = heisenberg_hamiltonian(lattice)

# Define how the Hamiltonian will be partitioned
# into a sum of sub-Hamiltonians.
function in_partition(sites::Tuple{Int,Int}, p, nparts)
  i, j = sites
  return p == mod1(i, nparts)
end

nprocs = MPI.Comm_size(MPI.COMM_WORLD)
ℋs = partition(ℋ, nprocs; in_partition=in_partition)

PH = if mpi_projmpo
  n = MPI.Comm_rank(MPI.COMM_WORLD) + 1
  MPISum(ProjMPO(MPO(ℋs[n], sites)))
else
  ProjMPOSum(H)
end

state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
psi0 = randomMPS(sites, state, 10)

sweeps = Sweeps(10)
setmaxdim!(sweeps, 20, 60, 100, 100, 200, 400, 800)
setcutoff!(sweeps, 1E-10)
@show sweeps

energy, psi = dmrg(PH, psi0, sweeps)
@show energy

MPI.Finalize()
