struct MPI_MPS{T<:Union{MPS,MPO}} <: AbstractMPS
  data::T
  left_bond::ITensor
  right_bond::ITensor
  comm::MPI.Comm
end

function global_orthogonalize(ψ::MPI_MPS, ortho_centers=ones(Int, MPI.Comm_size(ψ.comm)))

end

function MPI_MPS(ψ::AbstractMPS; orthogonalize=false, ortho_centers=ones(Int, MPI.Comm_size(ψ.comm)), comm::MPI.Comm=MPI.COMM_WORLD)
  mpi_size = MPI.Comm_size(comm)
  mpi_rank = MPI.Comm_rank(comm)
  n = length(ψ)
  # Split ψ into partitions
  partition_start = ceil(Int, n / mpi_size) * mpi_rank + 1
  partition_stop = min(ceil(Int, n / mpi_size) * (mpi_rank + 1), n)

  @show partition_start:partition_stop

  ψ_MPI = MPI_MPS(ψ[partition_start:partition_stop], ITensor(1.0), ITensor(1.0), comm)
  if orthogonalize
    ψ_MPI = global_orthogonalize(ψ_MPI, ortho_centers)
  end
  return ψ_MPI
end
