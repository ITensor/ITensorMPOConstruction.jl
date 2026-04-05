# ITensorMPOConstruction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/ITensorMPOConstruction.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/ITensorMPOConstruction.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorMPOConstruction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorMPOConstruction.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorMPOConstruction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorMPOConstruction.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![DOI](https://zenodo.org/badge/749825641.svg)](https://doi.org/10.5281/zenodo.17309443)

A fast algorithm for constructing a Matrix Product Operator (MPO) from a sum of local operators. This is a replacement for `ITensorMPS.MPO(os::OpSum, sites::Vector{<:Index})`. If julia is started with multiple threads, they are used to transparently speed up the construction.

The three goals of this library are

1. Produce exact MPOs (up to floating point error) of the smallest possible bond dimension.
2. Maximize the block sparsity of the resulting MPOs.
3. Accomplish these goals as fast as possible.

ITensorMPOConstruction is not designed to construct approximate compressed MPOs. If this is your workflow, use ITensorMPOConstruction to construct the exact MPO and call `ITensorMPS.truncate!`.

All runtimes below are taken from a single sample on a 2021 MacBook Pro with the M1 Max CPU and 32GB of memory.

## Installation

The package is currently not registered. Please install with the commands:
```julia
julia> using Pkg; Pkg.add(url="https://github.com/ITensor/ITensorMPOConstruction.jl.git")
```

## Citing

If you use this library in your research, please cite the following article https://doi.org/10.1103/nzrt-l2j1

## Questions or Issues

In addition to GitHub issues, you can ask question on the [ITensors discourse](https://itensor.discourse.group/).