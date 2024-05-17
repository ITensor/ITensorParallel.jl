using Compat
using ITensorParallel
using Test

@testset "Test $(@__FILE__)" begin
  examples_dir = joinpath(pkgdir(ITensorParallel), "examples")

  example_files = filter(
    f -> startswith(f, "03_") && endswith(f, ".jl"), readdir(examples_dir)
  )

  @testset "Hamiltonian partitioning example $example_file" for example_file in
                                                                example_files
    include(joinpath(examples_dir, example_file))

    nx = 16
    ny = 8
    @compat (; os, os_partition_manual, os_partition_auto) = main(nx, ny)

    @test length(os) == length(sum(os_partition_manual)) &&
      all(∈(os), sum(os_partition_manual))
    @test length(os) == length(sum(os_partition_auto)) && all(∈(os), sum(os_partition_auto))

    @test length(os_partition_auto) == 3 * (ny + 1)
    @test length(os_partition_manual) == length(os_partition_auto)

    s = siteinds("S=1/2", nx * ny)
    ψ = random_mps(s, j -> isodd(j) ? "↑" : "↓"; linkdims=10)

    H = MPO(os, s)
    @test maxlinkdim(H) == 3 * (ny + 1) - 1

    E = inner(ψ', H, ψ)

    H_partition_manual = [MPO(os, s) for os in os_partition_manual]
    @test maximum(maxlinkdim, H_partition_manual) == 3
    E_partition_manual = sum((inner(ψ', H, ψ) for H in H_partition_manual))
    @test E_partition_manual ≈ E

    H_partition_auto = [MPO(os, s) for os in os_partition_auto]
    @test maximum(maxlinkdim, H_partition_auto) == 3
    E_partition_auto = sum((inner(ψ', H, ψ) for H in H_partition_auto))
    @test E_partition_auto ≈ E
  end
end
