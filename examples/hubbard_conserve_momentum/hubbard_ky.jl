using ITensors

function hubbard_ky_hopping(; Nx::Int, Ny::Int, t=1.0)
  opsum = OpSum()
  for x in 0:(Nx - 1)
    for ky in 0:(Ny - 1)
      i = x * Ny + ky + 1
      ϵ = -2 * t * cos((2 * π / Ny) * ky)
      if abs(ϵ) > 1e-12
        opsum += ϵ, "n↑", i
        opsum += ϵ, "n↓", i
      end
    end
  end
  for x in 0:(Nx - 2)
    for ky in 0:(Ny - 1)
      i = x * Ny + ky + 1
      j = (x + 1) * Ny + ky + 1
      opsum -= t, "c†↑", i, "c↑", j
      opsum -= t, "c†↑", j, "c↑", i
      opsum -= t, "c†↓", i, "c↓", j
      opsum -= t, "c†↓", j, "c↓", i
    end
  end
  return opsum
end

function hubbard_ky_interactions(; Nx::Int, Ny::Int, U)
  opsums = OpSum[]
  for py in 1:Ny
    for qy in 1:Ny
      opsum = OpSum()
      for x in 1:Nx
        for ky in 1:Ny
          i = (x - 1) * Ny + mod1(ky + qy + Ny - 1, Ny)
          j = (x - 1) * Ny + mod1(py - qy + Ny + 1, Ny)
          k = (x - 1) * Ny + py
          l = (x - 1) * Ny + ky
          opsum += U / Ny, "c†↓", i, "c†↑", j, "c↑", k, "c↓", l
        end
      end
      push!(opsums, opsum)
    end
  end
  return opsums
end

function hubbard_ky(; Nx::Int, Ny::Int, t=1.0, U)
  opsum_hopping = hubbard_ky_hopping(; Nx, Ny, t)
  opsum_interactions = hubbard_ky_interactions(; Nx, Ny, U)
  return [opsum_hopping; opsum_interactions]
end

