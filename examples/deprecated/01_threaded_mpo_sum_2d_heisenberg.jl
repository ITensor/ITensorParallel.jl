using ITensors
using ITensorParallel
using Random

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

# Turn block sparse multithreading on or off
ITensors.disable_threaded_blocksparse()

npartitions = 2

# Threaded or sequential sum of MPOs
Sum = ThreadedSum
# Sum = SequentialSum

@show Threads.nthreads()
@show npartitions

Nx, Ny = 10, 5
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

ℋs = partition(ℋ, npartitions; in_partition)
H = [MPO(ℋ, sites) for ℋ in ℋs]
PH = Sum(H)

state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
Random.seed!(1234)
psi0 = randomMPS(sites, state; linkdims=10)

nsweeps = 10
maxdim = [20, 60, 100, 100, 200, 400, 800]
cutoff = 1e-6
@show nsweeps
@show maxdim
@show cutoff

energy, psi = dmrg(PH, psi0; nsweeps, maxdim, cutoff)
@show energy
