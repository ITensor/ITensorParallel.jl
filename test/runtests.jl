using ITensorParallel
using MPI
using Test

@testset "ITensorParallel.jl" begin
  test_files = filter(file -> startswith("test_")(file) && endswith(".jl")(file), readdir())
  @testset "Test file $(test_file)" for test_file in test_files
    include(test_file)
  end
end
