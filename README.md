# ITensorMPOConstruction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/ITensorMPOConstruction.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/ITensorMPOConstruction.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorMPOConstruction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorMPOConstruction.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorMPOConstruction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorMPOConstruction.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![DOI](https://zenodo.org/badge/749825641.svg)](https://doi.org/10.5281/zenodo.17309443)

A package for fast construction of Matrix Product Operators (MPOs) from sums of local operators. It provides an alternative to `ITensorMPS.MPO(os::OpSum, sites::Vector{<:Index})`.

The three goals of this library are

1. Produce exact MPOs (up to floating-point error) of the smallest possible bond dimension.
2. Maximize the block sparsity of the resulting MPOs.
3. Accomplish these goals as fast as possible.

ITensorMPOConstruction is not designed to construct approximate compressed MPOs. If this is your workflow, construct the exact MPO, then call `ITensorMPS.truncate!`.

## Installation

The package is registered and can be installed with the usual commands:
```julia
julia> using Pkg; Pkg.add("ITensorMPOConstruction")
```

## Algorithms

Currently, two different construction algorithms are supported:

1. A rank-decomposition algorithm based on the QR decomposition. It is guaranteed to produce MPOs of minimal bond dimension in all cases. This is the default, and can be selected explicitly with `MPO_new(...; alg="QR")`.

2. The minimum vertex cover algorithm from [RenLi2020](https://doi.org/10.1063/5.0018149). It is guaranteed to produce an MPO of minimal bond dimension **among all operators that share the same sparsity pattern**. In the cases where the vertex cover algorithm produces bond dimensions similar to the QR algorithm, the vertex cover construction is often much faster and produces MPOs of greater sparsity when `splitblocks=true`. Use it with `MPO_new(...; alg="VC")`.

## Citing

If you use this library in your research, please cite the following article [https://doi.org/10.1103/nzrt-l2j1](https://doi.org/10.1103/nzrt-l2j1).

## Questions or Issues

In addition to GitHub issues, you can ask questions on the [ITensors discourse](https://itensor.discourse.group/).
