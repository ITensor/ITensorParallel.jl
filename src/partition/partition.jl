# Generic interface for `partition`, used to partition
# the terms of an `OpSum` into a collection (`Vector`) of
# `OpSum`s, with the goal that the MPO representations
# will have smaller bond dimensions than the `MPO` representation
# of the original `OpSum`.
function partition(os::OpSum, args...; alg, kwargs...)
  return partition(Algorithm(alg), os, args...; kwargs...)
end

default_in_partition_alg() = "sum_split"

# Version where an algorithm is specified to determine
# which partition a term of the OpSum will go into
# based on the sites.
function partition(
  alg::Algorithm, os::OpSum, nparts::Integer; in_partition_alg=default_in_partition_alg()
)
  oss = [OpSum() for _ in 1:nparts]
  for p in 1:nparts
    for n in 1:length(os)
      sites = _sites(os[n])
      if in_partition(sites, p, nparts; in_partition_alg)
        add!(oss[p], os[n])
      end
    end
  end
  return oss
end

# A function returning the support of the term of an OpSum
# as a tuple.
function _sites(t)
  return Tuple((only(ITensors.site(o)) for o in ITensors.terms(t)))
end

_sort(t::Tuple; kwargs...) = typeof(t)(sort(collect(t); kwargs...))
