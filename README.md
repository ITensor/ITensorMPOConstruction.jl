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

## Reporting Issues: TODO

## Benchmarks: Fermi-Hubbard Hamiltonian in Momentum Space

We constructed the momentum space Fermi-Hubbard Hamiltonian using ITensorMPS and ITensorMPOConstruction. For even $N$, the Hamiltonian can be represented exactly as an MPO of bond dimension $10 N - 4$, and both algorithms achieve this minimal bond dimension. ITensorMPOConstruction is also not only able to construct this particular MPO much faster, but the sparsity of the resulting MPO is much higher.

### Bond Dimension 
| $N$ | ITensorMPS | ITensorMPOConstruction |
|-----|------------|------------------------|
| 10  | 96         | 96                     |
| 20  | 196        | 196                    |
| 30  | 296        | 296                    |
| 40  | N/A        | 396                    |
| 50  | N/A        | 496                    |
| 100 | N/A        | 996                    |
| 200 | N/A        | 1996                   |
| 300 | N/A        | 2996                   |
| 400 | N/A        | 3996                   |
| 500 | N/A        | 4996                   |

### Sparsity 

Sparsity of the `ITensorMPS` MPO with the default `splitblocks=true`, and the `ITensorMPOConstruction` MPO with the less aggressive `combine_qn_sectors::Bool=false`.

| $N$ | ITensorMPS | ITensorMPOConstruction |
|-----|------------|------------------------|
| 10  | 92.7%      | 99.32%                 |
| 20  | 92.6%      | 99.70%                 |
| 30  | 92.6%      | 99.81%                 |
| 40  | N/A        | 99.86%                 |
| 50  | N/A        | 99.89%                 |
| 100 | N/A        | 99.94%                 |
| 200 | N/A        | 99.97%                 |
| 300 | N/A        | 99.982%                |
| 400 | N/A        | 99.986%                |
| 500 | N/A        | 99.999%                |

### Runtime 
| $N$ | ITensorMPS | ITensorMPOConstruction |
|-----|------------|------------------------|
| 10  | 0.32s      | 0.009s                 |
| 20  | 30.6s      | 0.052s                 |
| 30  | 792s       | 0.14s                  |
| 40  | N/A        | 0.38s                  |
| 50  | N/A        | 0.63s                  |
| 100 | N/A        | 7.7s                   |
| 200 | N/A        | 103s                   |
| 300 | N/A        | 500s                   |
| 400 | N/A        | 1554s                  |
| 500 | N/A        | 3802s                  |
