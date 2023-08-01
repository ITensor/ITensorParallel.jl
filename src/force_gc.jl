# Default to 6 GB threshold to trigger GC
default_gc_gb_threshold() = 6.0
const gc_gb_threshold = Ref(default_gc_gb_threshold())
get_gc_gb_threshold() = gc_gb_threshold[]
function set_gc_gb_threshold!(gb_threshold)
  gc_gb_threshold[] = gb_threshold
  return nothing
end

# https://discourse.julialang.org/t/from-multithreading-to-distributed/101984/6
function force_gc(gb_threshold::Real=get_gc_gb_threshold())
  if Sys.free_memory() < gb_threshold * 2^30
     GC.gc()
  end
  return nothing
end
