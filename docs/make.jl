using ITensorParallel
using Documenter

DocMeta.setdocmeta!(ITensorParallel, :DocTestSetup, :(using ITensorParallel); recursive=true)

makedocs(;
    modules=[ITensorParallel],
    authors="Matthew Fishman <mfishman@flatironinstitute.org> and contributors",
    repo="https://github.com/mtfishman/ITensorParallel.jl/blob/{commit}{path}#{line}",
    sitename="ITensorParallel.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mtfishman.github.io/ITensorParallel.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mtfishman/ITensorParallel.jl",
    devbranch="main",
)
