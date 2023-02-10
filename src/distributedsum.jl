# Functionality for distributed terms of a sum
# onto seperate processes
function distribute(terms::Vector)
  return map(term -> @spawnat(:any, term), terms)
end

distribute(terms::Vector{Future}) = terms

function distribute(terms::Vector{MPO})
  return distribute(ProjMPO.(terms))
end

function DistributedSum(terms::Vector; executor_kwargs...)
  return FoldsSum(distribute(terms), SequentialEx(; executor_kwargs...))
end

function position!(term::Future, v::MPS, pos::Int)
  return @spawnat term.where position!(fetch(term), v, pos)
end

function product(term::Future, v::ITensor)
  return @fetchfrom term.where fetch(term)(v)
end
(term::Future)(v::ITensor) = product(term, v)

function noiseterm(term::Future, v::ITensor, dir::String)
  return @fetchfrom term.where noiseterm(fetch(term), v, dir)
end

function disk(term::Future; disk_kwargs...)
  return @spawnat term.where disk(fetch(term); disk_kwargs...)
end
