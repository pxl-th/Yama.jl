using Yama
using Documenter

makedocs(;
    modules=[Yama],
    authors="Anton Smirnov",
    repo="https://github.com/pxl-th/Yama.jl/blob/{commit}{path}#L{line}",
    sitename="Yama.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pxl-th.github.io/Yama.jl",
        assets=String[],
    ),
    pages=[
        "index.md",
        "User API" => "docs.md",
    ],
)
deploydocs(repo="github.com/pxl-th/Yama.jl")
