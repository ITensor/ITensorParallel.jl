using ITensors
using MPI
using ITensorParallel

let
    MPI.Init()
    N = 100
    sites = siteinds("S=1",N)

    ampo = OpSum()
    for j=1:N-1
        ampo += "Sz",j,"Sz",j+1
        ampo += 1/2,"S+",j,"S-",j+1
        ampo += 1/2,"S-",j,"S+",j+1
    end
    H = MPO(ampo,sites)

    psi0 = randomMPS(sites,10)
    
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 10,10)
    setcutoff!(sweeps, 1E-10)
    
    energy, psi = dmrg(H,psi0, sweeps)

    psi_parallel = ITensorParallel.PMPS(psi)

    MPI.Finalize()
end
    
