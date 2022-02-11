using ITensors
using ITensorParallel

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

# Turn block sparse multithreading on or off
ITensors.disable_threaded_blocksparse()

@show Threads.nthreads()

threaded_projmpo = true

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

ℋs = opsum_sum(ℋ, Threads.nthreads(); in_partition=in_partition)

H = Vector{MPO}(undef, Threads.nthreads())
Threads.@threads for n in 1:Threads.nthreads()
  H[Threads.threadid()] = splitblocks(linkinds, MPO(ℋs[n], sites))
end

PH = if threaded_projmpo
  ThreadedProjMPOSum(H)
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
