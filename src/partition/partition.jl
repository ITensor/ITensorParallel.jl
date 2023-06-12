# Generic interface for `partition`, used to partition
# the terms of an `OpSum` into a collection (`Vector`) of
# `OpSum`s, with the goal that the MPO representations
# will have smaller bond dimensions than the `MPO` representation
# of the original `OpSum`.
function partition(opsum::OpSum, args...; alg=nothing, kwargs...)
  return partition(Algorithm(alg), opsum, args...; kwargs...)
end

# Version where an algorithm is specified to determine
# which partition a term of the OpSum will go into
# based on the sites.
function partition(
  ::Algorithm,
  opsum::OpSum,
  npartitions;
  in_partition_alg=default_in_partition_alg(),
  in_partition=(sites, partition, npartitions) ->
    in_partition(sites, partition, npartitions; alg=in_partition_alg),
)
  opsums = [OpSum() for _ in 1:nparts]
  for partition in 1:npartitions
    for term_index in 1:length(opsum)
      term = opsum[term_index]
      sites = _sites(term)
      if in_partition(sites, partition, npartitions)
        add!(opsums[partition], term)
      end
    end
  end
  return opsums
end

function in_partition(sites, partition, npartitions; alg=default_in_partition_alg())
  return in_partition(Algorithm(alg), sites, partition, npartitions)
end

default_in_partition_alg() = Algorithm"sum_split"()

# A function returning the support of the term of an OpSum
# as a tuple.
function _sites(t)
  return Tuple((only(ITensors.site(o)) for o in ITensors.terms(t)))
end

# Sort a tuple and return a tuple.
_sort(t::Tuple; kwargs...) = typeof(t)(sort(collect(t); kwargs...))
