using Distributed

struct DistributedSum{T}
  data::Vector{T}
end

# Construct from a distributed sum from a
# function that returns the elements of the sum.
function DistributedSum(f::Function, n::Integer)
  # Assign the matrices to the first `n` workers.
  return DistributedSum([@spawnat(workers()[ni], f(ni)) for ni in 1:n])
end

# Functions needed for @distributed loop and reduction
Base.length(A::DistributedSum) = length(A.data)
Base.firstindex(A::DistributedSum) = firstindex(A.data)
Base.lastindex(A::DistributedSum) = lastindex(A.data)
Base.getindex(A::DistributedSum, args...) = getindex(A.data, args...)
Base.iterate(A::DistributedSum, args...) = iterate(A.data, args...)

# Apply the sum
function (A::DistributedSum)(v)
  return @distributed (+) for An in A
    fetch(An)(v)
  end
end
