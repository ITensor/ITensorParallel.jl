using ITensorParallel
using MPI
using Test

@testset "ITensorParallel.jl" begin
  examples_dir = joinpath(pkgdir(ITensorParallel), "examples")

  example_files = filter(
    f -> startswith(f, "01_") && endswith(f, ".jl"), readdir(examples_dir)
  )
  @testset "Threaded/Distributed example $example_file" for example_file in example_files
    include(joinpath(examples_dir, example_file))
    Nx = 8
    Ny = 4
    maxdim = 20
    Sums = (ThreadedSum, DistributedSum, SequentialSum)
    @testset "Sum type $Sum, threaded block sparse $threaded_blocksparse, write-to-disk $disk" for Sum in
                                                                                                   Sums,
      threaded_blocksparse in (false, true),
      disk in (false, true)

      println("Running parallel test with $(Sum)")
      main(; Nx, Ny, maxdim, Sum, disk, threaded_blocksparse)
    end
  end

  example_files = ["02_mpi_run.jl"]
  @testset "MPI example $example_file, threaded block sparse $threaded_blocksparse, write-to-disk $disk" for example_file in example_files, threaded_blocksparse in (false, true), disk in (false, true)
    println("Running MPI parallel test")
    nprocs = 2
    Nx = 8
    Ny = 4
    maxdim = 20
    mpiexec() do exe  # MPI wrapper
      run(
        `$exe -n $(nprocs) $(Base.julia_cmd()) $(joinpath(examples_dir, example_file)) --Nx $(Nx) --Ny $(Ny) --maxdim $(maxdim) --disk $(disk) --threaded_blocksparse $(threaded_blocksparse)`,
      )
    end
  end
end
