distribute(term) = @spawnat(:any, term)
distribute(term::Future) = term
distribute(term::MPO) = distribute(ProjMPO(term))

# Functionality for distributed terms of a sum
# onto seperate processes
distribute(terms::Vector) = distribute.(terms)

function DistributedSum(terms::Vector; executor_kwargs...)
  return FoldsSum(distribute(terms), SequentialEx(; executor_kwargs...))
end

(term::Future)(v::ITensor) = product(term, v)

function product(term::Future, v::ITensor)
  return product(() -> force_gc(), term, v)
end

function product(callback, term::Future, v::ITensor)
  return @fetchfrom term.where begin
    res = fetch(term)(v)
    callback()
    return res
  end
end

function position!(term::Future, v::MPS, pos::Int)
  return position!(() -> force_gc(), term, v, pos)
end

function position!(callback, term::Future, v::MPS, pos::Int)
  return @spawnat term.where begin
    res = position!(fetch(term), v, pos)
    callback()
    return res
  end
end

function noiseterm(term::Future, v::ITensor, dir::String)
  return noiseterm(() -> force_gc(), v, dir)
end

function noiseterm(callback, term::Future, v::ITensor, dir::String)
  return @fetchfrom term.where begin
    res = noiseterm(fetch(term), v, dir)
    callback()
    return res
  end
end

function disk(term::Future; disk_kwargs...)
  return disk(() -> force_gc(), term; disk_kwargs...)
end

function disk(callback, term::Future; disk_kwargs...)
  return @spawnat term.where begin
    res = disk(fetch(term); disk_kwargs...)
    callback()
    return res
  end
end
