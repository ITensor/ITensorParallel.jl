using ITensors
using ITensorParallel

function heisenberg_2d(nx, ny)
  lattice = square_lattice(nx, ny; yperiodic=false)
  os = OpSum()
  for b in lattice
    os += 0.5, "S+", b.s1, "S-", b.s2
    os += 0.5, "S-", b.s1, "S+", b.s2
    os += "Sz", b.s1, "Sz", b.s2
  end
  return os
end

function heisenberg_2d_grouped(nx, ny)
  os_grouped_1 = OpSum[]
  os_grouped_2 = OpSum[]
  os_grouped_3 = OpSum[]

  # Horizontal terms
  for jy in 1:ny
    os_1 = OpSum()
    os_2 = OpSum()
    os_3 = OpSum()
    for jx in 1:(nx - 1)
      j1, j2 = (jx - 1) * ny + jy, jx * ny + jy
      os_1 += 0.5, "S+", j1, "S-", j2
      os_2 += 0.5, "S-", j1, "S+", j2
      os_3 += "Sz", j1, "Sz", j2
    end
    push!(os_grouped_1, os_1)
    push!(os_grouped_2, os_2)
    push!(os_grouped_3, os_3)
  end

  # Vertical terms
  os_1 = OpSum()
  os_2 = OpSum()
  os_3 = OpSum()
  for jx in 1:nx
    for jy in 1:(ny - 1)
      j1, j2 = (jx - 1) * ny + jy, (jx - 1) * ny + jy + 1
      os_1 += 0.5, "S+", j1, "S-", j2
      os_2 += 0.5, "S-", j1, "S+", j2
      os_3 += "Sz", j1, "Sz", j2
    end
  end
  push!(os_grouped_1, os_1)
  push!(os_grouped_2, os_2)
  push!(os_grouped_3, os_3)

  # Remove empty OpSums
  os_grouped_1 = filter(!isempty, os_grouped_1)
  os_grouped_2 = filter(!isempty, os_grouped_2)
  os_grouped_3 = filter(!isempty, os_grouped_3)
  return [os_grouped_1; os_grouped_2; os_grouped_3]
end

function main(nx, ny)
  os = heisenberg_2d(nx, ny)
  os_partition_manual = heisenberg_2d_grouped(nx, ny)
  os_partition_auto = partition(os; alg="chain_split")
  return (; os, os_partition_manual, os_partition_auto)
end
