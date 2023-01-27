# 02_mpi_run.jl
include("02_mpi_mpo_sum_2d_hubbard_conserve_momentum.jl")

# Run with:
# mpiexecjl -n 2 julia 02_mpi_run.jl --Nx 8 --Ny 4 --maxdim 20

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
    "--maxdim"
      help = "Maximum bond dimension"
      arg_type = Int
      required = true
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
  maxdim=args["maxdim"],
  threaded_blocksparse=args["threaded_blocksparse"],
);
