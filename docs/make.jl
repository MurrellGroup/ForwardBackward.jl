using ForwardBackward
using Documenter

DocMeta.setdocmeta!(ForwardBackward, :DocTestSetup, :(using ForwardBackward); recursive=true)

makedocs(;
    modules=[ForwardBackward],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="ForwardBackward.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/ForwardBackward.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/ForwardBackward.jl",
    devbranch="main",
)
