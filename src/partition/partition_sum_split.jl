# Algorithm for splitting an `OpSum` proposed by Zhai and Chan in https://arxiv.org/abs/2103.09976.
function in_partition(
  ::Algorithm"sum_split", sites::Tuple{Int}, p::Integer, nparts::Integer
)
  return p == mod1(sites[1], nparts)
end

function in_partition(::Algorithm"sum_split", sites::NTuple{2,Int}, p, nparts)
  # Make sure `i ≥ j`
  i, j = _sort(sites; rev=true)
  return p == mod1((i - 1) * i ÷ 2 + j, nparts)
end

function in_partition_sorted(alg::Algorithm"sum_split", sites::NTuple{4,Int}, p, nparts)
  i, j, k, l = sites
  sites = j == k ? (j,) : (i, j)
  return in_partition(alg, sites, p, nparts)
end

function in_partition(alg::Algorithm"sum_split", sites::NTuple{4,Int}, p, nparts)
  return in_partition_sorted(alg, _sort(sites), p, nparts)
end
