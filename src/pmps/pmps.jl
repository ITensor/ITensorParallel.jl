"""
    PMPS

A finite size matrix product state type for distributed parallel computations
Keeps track of the orthogonality center.
"""
mutable struct PMPS <: ITensors.AbstractMPS
  data::Vector{ITensor}
  llim::Int
  rlim::Int
  comm::MPI.Comm
end


function PMPS(psi0::MPS; comm::MPI.Comm=MPI.COMM_WORLD)

    psi = copy(psi0)
    
    mpi_size = MPI.Comm_size(comm)
    mpi_rank = MPI.Comm_rank(comm)

    println("hello from $(mpi_rank) of $(mpi_size)")
    N = length(psi)
    data_index_begin = Int(ceil(float(N) /  float(mpi_size))) * mpi_rank + 1
    data_index_end = min(N, Int(ceil(float(N) /  float(mpi_size))) * (mpi_rank + 1))

    orthogonalize!(psi, data_index_begin)

    data = copy(psi.data[data_index_begin:data_index_end])
    
    println("#$(mpi_rank): $(data_index_begin) $(data_index_end)")

    return PMPS(data, psi.llim, psi.rlim, comm)
end
