using ITensorMPOConstruction
using ITensors
using ITensorMPS
using Documenter
using Literate

DocMeta.setdocmeta!(
  ITensorMPOConstruction, :DocTestSetup, :(using ITensorMPOConstruction); recursive=true
)

function preprocess(content)
  return replace(content, r"#START_HIDE.*?#END_HIDE"s => "\1s#src")
end

examples = [
  "./examples/fermi-hubbard-rs.jl",
  "./examples/fermi-hubbard-ks.jl",
  "./examples/haldane-shastry.jl",
  "./examples/electronic-structure.jl",
  "./examples/fermi-hubbard-tc.jl",
]

for example in examples
  Literate.markdown(
    example,
    "./docs/src/examples/";
    flavor=Literate.CommonMarkFlavor(),
    # preprocess
    execute=false,
  )
end

cp("./README.md", "./docs/src/index.md"; force=true)

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
    "Documentation" => [
      "MPO_new" => "documentation/MPO_new.md",
      "OpIDSum" => "documentation/OpID.md",
      "Internal Functions" => "documentation/unexported.md",
    ],
    "Algorithm Selection" => "algorithm-selection.md",
    "Examples" => [
      "Real Space Fermi-Hubbard" => "examples/fermi-hubbard-rs.md",
      "Momentum Space Fermi-Hubbard" => "examples/fermi-hubbard-ks.md",
      "Electronic Structure" => "examples/electronic-structure.md",
      "Haldane-Shastry and Truncation" => "examples/haldane-shastry.md",
      "Challenge Problem" => "examples/fermi-hubbard-tc.md",
    ],
    "Threading and Performance" => "threading.md",
  ],
)

deploydocs(;
  repo="github.com/ITensor/ITensorMPOConstruction.jl", devbranch="main", push_preview=true
)
