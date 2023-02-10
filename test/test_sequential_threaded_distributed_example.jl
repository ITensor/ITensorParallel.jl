using ITensorParallel
using Test

@testset "Test $(@__FILE__)" begin
  examples_dir = joinpath(pkgdir(ITensorParallel), "examples")

  example_files = filter(
    f -> startswith(f, "01_") && endswith(f, ".jl"), readdir(examples_dir)
  )
  @testset "Threaded/Distributed example $example_file" for example_file in example_files
    include(joinpath(examples_dir, example_file))
    Nx = 8
    Ny = 4
    maxdim = 20
    Sums = (SequentialSum, ThreadedSum, DistributedSum)
    @testset "Sum type $Sum, threaded block sparse $threaded_blocksparse, write-to-disk $disk" for Sum in
                                                                                                   Sums,
      threaded_blocksparse in (false, true),
      disk in (false, true)

      println(
        "\nRunning parallel test with $(Sum), threaded block sparse $threaded_blocksparse, write-to-disk $disk",
      )
      main(; Nx, Ny, maxdim, Sum, disk, threaded_blocksparse)
    end
  end
end
