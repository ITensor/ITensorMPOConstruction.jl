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

ITensorMPOConstruction is not designed to construct approximate compressed MPOs. If this is your workflow, construct the exact MPO then call `ITensorMPS.truncate!`.

## Installation

The package is registered and can be installed with the usual commands:
```julia
julia> using Pkg; Pkg.add("ITensorMPOConstruction")
```

## Citing

If you use this library in your research, please cite the following article [https://doi.org/10.1103/nzrt-l2j1](https://doi.org/10.1103/nzrt-l2j1).

## Questions or Issues

In addition to GitHub issues, you can ask question on the [ITensors discourse](https://itensor.discourse.group/).