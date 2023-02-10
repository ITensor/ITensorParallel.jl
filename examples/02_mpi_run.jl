# 02_mpi_run.jl
include("02_mpi_mpo_sum_2d_hubbard_conserve_momentum.jl")

# Run with:
# mpiexecjl -n 2 julia 02_mpi_run.jl --Nx 8 --Ny 4 --nsweeps 10 --maxdim 1000

using ArgParse
function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--Nx"
    help = "Cylinder length"
    arg_type = Int
    required = true
    "--Ny"
    help = "Cylinder width"
    arg_type = Int
    required = true
    "--nsweeps"
    help = "Number of sweeps"
    arg_type = Int
    required = true
    "--maxdim"
    help = "Maximum bond dimension"
    arg_type = Int
    required = true
    "--disk"
    help = "Write-to-disk"
    arg_type = Bool
    default = false
    "--threaded_blocksparse"
    help = "Use threaded block sparse operations"
    arg_type = Bool
    default = false
  end
  return parse_args(s)
end
args = parse_commandline()
main(;
  Nx=args["Nx"],
  Ny=args["Ny"],
  nsweeps=args["nsweeps"],
  maxdim=args["maxdim"],
  disk=args["disk"],
  threaded_blocksparse=args["threaded_blocksparse"],
);
