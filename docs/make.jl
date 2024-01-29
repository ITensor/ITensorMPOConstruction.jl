using ITensorMPOConstruction
using Documenter

DocMeta.setdocmeta!(ITensorMPOConstruction, :DocTestSetup, :(using ITensorMPOConstruction); recursive=true)

makedocs(;
    modules=[ITensorMPOConstruction],
    authors="Ben Corbett and contributors",
    sitename="ITensorMPOConstruction.jl",
    format=Documenter.HTML(;
        canonical="https://ITensor.github.io/ITensorMPOConstruction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ITensor/ITensorMPOConstruction.jl",
    devbranch="main",
)
