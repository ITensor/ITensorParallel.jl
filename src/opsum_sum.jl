function _sites(t::ITensors.MPOTerm)
  return Tuple((only(o.site) for o in t.ops))
end

function opsum_sum(os::OpSum, nparts::Integer; in_partition=default_in_partition)
  oss = [OpSum() for _ in 1:nparts]
  for p in 1:nparts
    for n in 1:length(os)
      sites = _sites(os[n])
      if in_partition(sites, p, nparts)
        #oss[p] += os[n]
        push!(oss[p], os[n])
      end
    end
  end
  return oss
end
