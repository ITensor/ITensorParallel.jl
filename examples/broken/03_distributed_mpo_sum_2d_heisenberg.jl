using Distributed
nprocs = 2
addprocs(nprocs)
@show Threads.nthreads()

@everywhere using ITensors
@everywhere using ITensorParallel

@everywhere ITensors.BLAS.set_num_threads(1)
@everywhere ITensors.Strided.disable_threads()

# Turn block sparse multithreading on or off
@everywhere ITensors.disable_threaded_blocksparse()

distributed_projmpo = true

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

ℋs = partition(ℋ, nprocs; in_partition=in_partition)

PH = if distributed_projmpo
  DistributedSum(n -> ProjMPO(MPO(ℋs[n], sites)), nprocs)
else
  ProjMPOSum(H)
end

state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
psi0 = randomMPS(sites, state; linkdims=10)

nsweeps = 10
maxdims = [20, 60, 100, 100, 200, 400, 800]
cutoff = 1e-10
@show nsweeps
@show maxdims
@show cutoff

energy, psi = dmrg(PH, psi0; nsweeps, maxdims, cutoff)
@show energy
