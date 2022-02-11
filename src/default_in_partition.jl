function default_in_partition(sites::Tuple{Int}, p::Integer, nparts::Integer)
  return p == mod1(sites[1], nparts)
end

# i ≥ j
_f(i, j) = (i - 1) * i ÷ 2 + j
function default_in_partition(sites::NTuple{2,Int}, p, nparts)
  i, j = sites
  if i ≤ j
    return p == mod1(_f(j, i), nparts)
  end
  return p == mod1(_f(i, j), nparts)
end

function default_in_partition_sorted(sites::NTuple{4,Int}, p, nparts)
  i, j, k, l = sites
  if j == k
    return default_in_partition((j,), p, nparts)
  end
  return default_in_partition((i, j), p, nparts)
end

function default_in_partition(sites::NTuple{4,Int}, p, nparts)
  return default_in_partition_sorted(sort(sites), p, nparts)
end
