# Algorithm for splitting an `OpSum` proposed by Zhai and Chan in https://arxiv.org/abs/2103.09976.
function in_partition(::Algorithm"sum_split", sites::Tuple{Int}, partition, npartitions)
  return partition == mod1(sites[1], npartitions)
end

function in_partition(::Algorithm"sum_split", sites::NTuple{2,Int}, partition, npartitions)
  # Make sure `i ≥ j`
  i, j = _sort(sites; rev=true)
  return partition == mod1((i - 1) * i ÷ 2 + j, npartitions)
end

function in_partition_sorted(
  alg::Algorithm"sum_split", sites::NTuple{4,Int}, partition, npartitions
)
  i, j, k, l = sites
  sites = j == k ? (j,) : (i, j)
  return in_partition(alg, sites, partition, npartitions)
end

function in_partition(
  alg::Algorithm"sum_split", sites::NTuple{4,Int}, partition, npartitions
)
  return in_partition_sorted(alg, _sort(sites), partition, npartitions)
end
