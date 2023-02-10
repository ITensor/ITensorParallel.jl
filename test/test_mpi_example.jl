using MPI
using ITensorParallel
using Test

@testset "Test $(@__FILE__)" begin
  examples_dir = joinpath(pkgdir(ITensorParallel), "examples")
  example_files = ["02_mpi_run.jl"]
  @testset "MPI example $example_file, threaded block sparse $threaded_blocksparse, write-to-disk $disk" for example_file in
                                                                                                             example_files,
    threaded_blocksparse in (false, true),
    disk in (false, true)

    println(
      "\nRunning MPI parallel test with threaded block sparse $threaded_blocksparse, write-to-disk $disk",
    )
    nprocs = 2
    Nx = 8
    Ny = 4
    nsweeps = 2
    maxdim = 20
    mpiexec() do exe  # MPI wrapper
      run(
        `$exe -n $(nprocs) $(Base.julia_cmd()) --threads $(Threads.nthreads()) $(joinpath(examples_dir, example_file)) --Nx $(Nx) --Ny $(Ny) --nsweeps $(nsweeps) --maxdim $(maxdim) --disk $(disk) --threaded_blocksparse $(threaded_blocksparse)`,
      )
    end
  end
end
