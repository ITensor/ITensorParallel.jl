using ITensorParallel
using Test

@testset "ITensorParallel.jl" begin
  examples_dir = joinpath(pkgdir(ITensorParallel), "examples")
  # example_files = filter(endswith(".jl"), readdir(examples_dir))
  example_files = ["01_parallel_mpo_sum_2d_hubbard_conserve_momentum.jl"]
  @testset "Example $example_file" for example_file in example_files
    include(joinpath(examples_dir, example_file))
    maxdim = 20
    Sums = (ThreadedSum, DistributedSum, SequentialSum)
    @testset "Sum type $Sum" for Sum in Sums
      main(; maxdim, Sum)
    end
  end
end
