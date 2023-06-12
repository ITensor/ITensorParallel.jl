# Algorithm for splitting an `OpSum` into chains,
# suggested by Steve White in discussion with Matt Fishman
# 06/12/2023.
function partition(::Algorithm"chain_split", os::OpSum)
  chains = OpSum[]
  group_terms_into_chains!(chains, os)
  chains = merge_disjoint_chains(chains)
  return chains
end

# Return a chain group and the original OpSum with that
# group removed.
function chain_group_and_complement(os::OpSum)
  if isempty(os)
    return OpSum(), os
  end

  chain_terms_complement = copy(ITensors.terms(os))
  chain_supports_complement = ITensors.sites.(chain_terms_complement)
  sort!.(unique!.(chain_supports_complement))
  chain_terms = eltype(chain_terms_complement)[]
  chain_supports = eltype(chain_supports_complement)[]

  # Pick the first term.
  term_index = 1

  # Track the support of the chain.
  # Keep the sites in the support sorted.
  chain_support = Int[]

  while !isnothing(term_index)
    term = chain_terms_complement[term_index]
    term_support = chain_supports_complement[term_index]

    # Add the term to the chain and remove from the complement.
    push!(chain_terms, term)
    deleteat!(chain_terms_complement, term_index)
    deleteat!(chain_supports_complement, term_index)

    # Add the support of the term to the total support
    # of the chain.
    append!(chain_support, term_support)
    unique!(chain_support)

    # Search for the next term to add to the chain.
    term_index = findfirst(chain_supports_complement) do term_support
      # return minimum(ITensors.sites(term)) == last(chain_support)
      return first(term_support) == last(chain_support)
    end
  end
  return Sum(chain_terms), Sum(chain_terms_complement)
end

function group_terms_into_chains!(chains::Vector{<:OpSum}, os::OpSum)
  chain, chain_complement = chain_group_and_complement(os)
  if isempty(chain)
    # TODO: Is this check needed?
    if !isempty(chain_complement)
      push!(chains, chain_complement)
    end
    return chains
  end
  push!(chains, chain)
  group_terms_into_chains!(chains, chain_complement)
  return chains
end

# Merge non-overlapping/non-intertwining chains.
# The can be trivially combined without increasing
# the bond dimension of the MPO.
function merge_disjoint_chains(chains)
  if isempty(chains) || isone(length(chains))
    return chains
  end

  # Sort the chains by their support lexicographically.
  supports = ITensors.sites.(chains)
  sort!.(unique!.(supports))
  p = sortperm(supports)
  supports = supports[p]
  chains = chains[p]

  merged_chains = empty(chains)
  merged_supports = empty(supports)
  for chain_index in eachindex(chains)
    chain = chains[chain_index]
    support = supports[chain_index]
    merged_chain_index = findfirst(merged_supports) do merged_support
      return first(support) > last(merged_support)
    end
    if isnothing(merged_chain_index)
      push!(merged_chains, chain)
      push!(merged_supports, support)
    else
      merged_chains[merged_chain_index] += chain
      append!(merged_supports[merged_chain_index], support)
      sort!(unique!(merged_supports[merged_chain_index]))
    end
  end
  return merged_chains
end
