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

# How to split the terms across the threads
function proc(p, ptot, i)
  return p == mod1(i, ptot)
end

ℋ = [OpSum() for _ in 1:Threads.nthreads()]
for p in 1:Threads.nthreads()
  for b in lattice
    pj = proc(p, Threads.nthreads(), b.s1)
    if pj ≠ 0
      ℋ[p] .+= pj * 0.5, "S+", b.s1, "S-", b.s2
      ℋ[p] .+= pj * 0.5, "S-", b.s1, "S+", b.s2
      ℋ[p] .+= pj, "Sz", b.s1, "Sz", b.s2
    end
  end
end

H = Vector{MPO}(undef, Threads.nthreads())
Threads.@threads for n in 1:Threads.nthreads()
  H[Threads.threadid()] = splitblocks(linkinds, MPO(ℋ[n], sites))
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
