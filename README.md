# ITensorMPOConstruction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/ITensorMPOConstruction.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/ITensorMPOConstruction.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorMPOConstruction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorMPOConstruction.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorMPOConstruction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorMPOConstruction.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

A fast algorithm for constructing a Matrix Product Operator (MPO) from a sum of local operators. This is a replacement for `ITensorMPS.MPO(os::OpSum, sites::Vector{<:Index})`. If julia is started with multiple threads, they are used to transparently speed up the construction.

The three goals of this library are

1. Produce exact MPOs (up to floating point error) of the smallest possible bond dimension.
2. Maximize the block sparsity of the resulting MPOs.
3. Accomplish these goals as fast as possible.

ITensorMPOConstruction is not designed to construct approximate compressed MPOs. If this is your workflow, use ITensorMPOConstruction to construct the exact MPO and call `ITensorMPS.truncate!`.

## Installation

The package is currently not registered. Please install with the commands:
```julia
julia> using Pkg; Pkg.add(url="https://github.com/ITensor/ITensorMPOConstruction.jl.git")
```

## Constraints

This algorithm shares the same constraints as the default algorithm from ITensorMPS.

1. The operator must be expressed as a sum of products of single site operators. For example a CNOT could not appear in the sum since it is a two site operator.
2. When dealing with Fermionic systems the parity of each term in the sum must be even. That is the combined number of creation and annihilation operators in each term in the sum must be even. It should be possible to relax this constraint.

There are also two additional constraints:

3. Each term in the sum of products representation can only have a single operator acting on a site. For example a term such as $\mathbf{X}^{(1)} \mathbf{X}^{(1)}$ is not allowed. However, there is a pre-processing utility that can automatically replace $\mathbf{X}^{(1)} \mathbf{X}^{(1)}$ with $\mathbf{I}^{(1)}$. This is not a hard requirement for the algorithm but a simplification to improve performance.
4. When constructing a quantum number conserving operator the total flux of the operator must be zero. It would be easy to remove this constraint.

## `MPO_new`

The main exported function is `MPO_new` which takes an `OpSum` and transforms it into a MPO.

```julia
function MPO_new(os::OpSum, sites::Vector{<:Index}; kwargs...)::MPO
```

The optional keyword arguments are
* `basis_op_cache_vec`: A list of operators to use as a basis for each site. The operators on each site are expressed as one of these basis operators.
* `check_for_errors::Bool`: Check the input OpSum for errors, this can be expensive for larger problems.
* `tol::Real=1`: A multiplicative modifier to the default tolerance used in the SPQR, see [SPQR user guide Section 2.3](https://fossies.org/linux/SuiteSparse/SPQR/Doc/spqr_user_guide.pdf). The value of the default tolerance depends on the input matrix, which means a different tolerance is used for each decomposition. In the cases we have examined, the default tolerance works great for producing accurate MPOs.
* `absolute_tol::Bool=false`: Override the default adaptive tolerance scheme outlined above, and use the value of `tol` as the single tolerance for each decomposition.
* `combine_qn_sectors::Bool=false`: When `true`, the blocks of the MPO corresponding to the same quantum numbers are merged together into a single block. This can decrease the resulting sparsity.
* `call_back::Function=(cur_site::Int, H::MPO, sites::Vector{<:Index}, llinks::Vector{<:Index},cur_graph::MPOGraph, op_cache_vec::OpCacheVec) -> nothing`: A function that is called after constructing the MPO tensor at `cur_site`. Primarily used for writing checkpoints to disk for massive calculations.
* `output_level::Int=0`: Specify the amount of output printed to standard out, larger values produce more output. 

## Examples: Fermi-Hubbard Hamiltonian in Real Space

The one dimensional Fermi-Hubbard Hamiltonian with periodic boundary conditions on $N$ sites can be expressed in real space as

$$
\mathcal{H} = -t \sum_{i = 1}^N \sum_{\sigma \in (\uparrow, \downarrow)} \left( c^\dagger_{i, \sigma} c_{i + 1, \sigma} + c^\dagger_{i, \sigma} c_{i - 1, \sigma} \right) + U \sum_{i = 1}^N n_{i, \uparrow} n_{i, \downarrow} \ ,
$$

where the periodic boundary conditions enforce that $c_k = c_{k + N}$. For this Hamiltonian all that needs to be done to switch over to using ITensorMPOConstruction is switch `MPO(os, sites)` to `MPO_New(os, sites)`.

https://github.com/ITensor/ITensorMPOConstruction.jl/blob/637dd2409f27ede41aa916822ea8acb4cd557a9e/examples/fermi-hubbard.jl#L4-L24

For $N = 1000$ both ITensorMPS and ITensorMPOConstruction can construct an MPO of bond dimension 10 in under two seconds. When quantum number conservation is enabled, ITensorMPS produces an MPO that is 93.4% block sparse, whereas ITensorMPOConstruction's MPO is 97.4% block sparse.

## Examples: Fermi-Hubbard Hamiltonian in Momentum Space

The one dimensional Fermi-Hubbard Hamiltonian with periodic boundary conditions on $N$ sites can be expressed in momentum space as

$$
\mathcal{H} = \sum_{k = 1}^N \epsilon(k) \left( n_{k, \downarrow} + n_{k, \uparrow} \right) + \frac{U}{N} \sum_{p, q, k = 1}^N c^\dagger_{p - k, \uparrow} c^\dagger_{q + k, \downarrow} c_{q, \downarrow} c_{p, \uparrow}
$$

where $\epsilon(k) = -2 t \cos(\frac{2 \pi k}{N})$ and $c_k = c_{k + N}$. The code to construct the `OpSum` is shown below.

https://github.com/ITensor/ITensorMPOConstruction.jl/blob/637dd2409f27ede41aa916822ea8acb4cd557a9e/examples/fermi-hubbard.jl#L26-L49

Unlike the previous example, some more involved changes will be required to use ITensorMPOConstruction. This is because the `OpSum` has multiple operators acting on the same site, violating constraint #3. For example when $k = 0$ in the second loop we have terms of the form $c^\dagger_{p, \uparrow} c^\dagger_{q, \downarrow} c_{q, \downarrow} c_{p, \uparrow}$. You could always create a special case for $k = 0$ and rewrite it as $n_{p, \uparrow} n_{q, \downarrow}$. However if using "Electron" sites then you would also need to consider other cases such as when $p = q$, this would introduce a lot of extraneous complication. Luckily ITensorMPOConstruction provides a method to automatically perform these transformations. If you provide a set of operators for each site to `MPO_new` it will attempt to express the operators acting on each site as a single one of these "basis" operators. The code to do this is shown below.

https://github.com/ITensor/ITensorMPOConstruction.jl/blob/637dd2409f27ede41aa916822ea8acb4cd557a9e/examples/fermi-hubbard.jl#L51-L76

With $N = 20$ and quantum number conservation turned on, ITensorMPS produces an MPO of bond dimension 196 which is 92.6% sparse in 30s, whereas ITensorMPOConstruction produces an MPO of equal bond dimension but which is 99.7% sparse in 0.1s.

### `OpIDSum`

For $N = 200$ constructing the `OpSum` takes 42s and constructing the MPO from the `OpSum` with ITensorMPOConstruction takes another 306s. For some systems constructing the `OpSum` can actually be the bottleneck. In these cases you can construct an `OpIDSum` instead.

`OpIDSum` plays the same roll as `OpSum` but in a much more efficient manner. To specify an operator in a term of an `OpSum` you specify a string (the operator's name) and a site index, whereas to specify an operator in a term of an `OpIDSum` you specify an `OpID` which contains an operator index and a site. The operator index is the index of the operator in the provided basis for the site.

For $N = 200$ constructing an `OpIDSum` takes only 0.4s. Shown below is code to construct the Hamiltonian using an `OpIDSum`.

https://github.com/ITensor/ITensorMPOConstruction.jl/blob/637dd2409f27ede41aa916822ea8acb4cd557a9e/examples/fermi-hubbard.jl#L79-L130

Unlike `OpSum`, the `OpIDSum` constructor takes a few important arguments

```julia
function OpIDSum{N, C, Ti}(
  max_terms::Int,
  op_cache_vec::OpCacheVec;
  abs_tol::Real=0
)::OpIDSum{N, C, Ti} where {N, C, Ti}
```
* `N`: The maximum number of local operators appearing in any individual term. For the real space Fermi-Hubbard Hamiltonian `N = 2`, but in momentum space `N = 4`.
* `C`: The scalar weight type.
* `Ti`: The integer type used to index both the local operator basis and the number of sites. For example with `Ti = UInt8`, the local operator basis can have up to 255 elements and 255 sites can be used. Much of the memory consumption comes storing elements of type `Ti`.
* `max_terms`: The maximum number of terms in the `OpIDSum`, space is pre-allocated.
* `op_cache_vec`: Maps each `OpID` and site to a physical operator.
* `abs_tol`: Drop terms that have a weight of absolute value less than this.

An `OpIDSum` is constructed similarly to an `OpSum`, with support for the following two **threadsafe** functions for adding a term.

```julia
function ITensorMPS.add!(os::OpIDSum, scalar::Number, ops)::Nothing
function ITensorMPS.add!(os::OpIDSum, scalar::Number, ops::OpID...)::Nothing
```

Additionally, a further constructor is provided which takes in a modifying `function modify!(ops)::C` which is called as each term is added to the sum. It accepts a list of the sorted `OpID`, which it can modify, and returns a scalar which multiplies the weight. This is for advanced usage only, but in certain cases can greatly speed up and or simplify construction.

## Haldane-Shasty Hamiltonian and truncation

The Haldane-Shasty Hamiltonian defined on $N$ spin-$\frac{1}{2}$ particles is

$$
H = \frac{J \pi^2}{N^2} \sum_{n = 1}^N \sum_{m = n + 1}^N \frac{\mathbf{S}_m \cdot \mathbf{S}_n}{\sin^2 \left( \pi \frac{n - m}{N} \right)} \ .
$$

With $N = 40$ and using the default arguments, ITensorMPOConstruction creates an MPO of bond dimension 62, whereas ITensorMPS creates an MPO of bond dimension 38. However, this does not necessarily mean that ITensorMPS produces a better MPO. Comparing MPOs directly is tricky, but since the Haldane-Shasty Hamiltonian is exactly solvable, we can compare the energies and variances of the ground states obtained with DMRG for each MPO.

Using the MPO from ITensorMPOConstruction we obtain an error of $\delta = E_\text{DMRG} - E_\text{gs} = 3 \times 10^{-12}$ and a variance of $\sigma^2 = \braket{\psi_\text{DMRG} | (H - E_\text{DMRG})^2 | \psi_\text{DMRG}} = 2 \times 10^{-11}$, whereas with the MPO from ITensors the error increases to $\delta = 10^{-8}$ while the variance remains unchanged. The fact that the error in the energy increased while the variance remained constant suggests that ITensorMPS is performing a slightly lossy compression the Hamiltonian.

ITensorMPOConstruction is designed to construct exact MPOs (up to numerical precision), nevertheless, we can abuse it to perform approximate MPO construction. By setting `tol = 2E10` we obtain an MPO of bond dimension 38, equal to that produced by TensorMPS. However, using this approximate MPO we obtain poor results, with errors of $\delta = 10^{-3}$ and $\sigma^2 = 8 \times 10^{-9}$. The fact that such a high tolerance was required to reduce the bond dimension is a sign that this is not a good way of doing things. Setting `absolute_tol = true` to use a uniform cutoff across QR decompositions does not help either.

Starting with the MPO from ITensorMPOConstruction obtained with the standard `tol = 1` and then truncating down to a bond dimension of 38 using `ITensorMPS.truncate` yields DMRG errors of $\delta = 4 \times 10^{-9}$ and $\sigma^2 = 2 \times 10^{-11}$, better than those obtained with the MPO from ITensorMPS.

## Benchmarks: Fermi-Hubbard Hamiltonian in Momentum Space

All runtimes below are taken from a single sample on a 2021 MacBook Pro with the M1 Max CPU and 32GB of memory.

We constructed the momentum space Fermi-Hubbard Hamiltonian using ITensorMPS, ITensorMPOConstruction and [block2](https://github.com/block-hczhai/block2-preview) which has one of the most sophisticated MPO construction algorithms.

TODO: Once block2 results are in fill in comments

<!-- For even $N$, the Hamiltonian can be represented exactly as an MPO of bond dimension $10 N - 4$, and all the algorithms achieve this minimal bond dimension. -->

### Bond Dimension 
| $N$ | ITensorMPS | block2 | ITensorMPOConstruction |
|-----|------------|--------|------------------------|
| 10  | 96         | 96     | 96                     |
| 20  | 196        | 196    | 196                    |
| 30  | 296        | 296    | 296                    |
| 40  | N/A        | 396    | 396                    |
| 50  | N/A        | 496    | 496                    |
| 100 | N/A        | 996    | 996                    |
| 200 | N/A        | 1996   | 1996                   |
| 300 | N/A        | 2996   | 2996                   |
| 400 | N/A        | 3996   | 3996                   |
| 500 | N/A        | ????   | 4996                   |

### Sparsity 
| $N$ | ITensorMPS | block2 | ITensorMPOConstruction |
|-----|------------|--------|------------------------|
| 10  | 92.7%      | 96.7%  | 99.32%                 |
| 20  | 92.6%      | 98.4%  | 99.70%                 |
| 30  | 92.6%      | 99.0%  | 99.81%                 |
| 40  | N/A        | 99.2%  | 99.86%                 |
| 50  | N/A        | 99.4%  | 99.89%                 |
| 100 | N/A        | 99.7%  | 99.94%                 |
| 200 | N/A        | 99.85% | 99.97%                 |
| 300 | N/A        | 99.90% | 99.982%                |
| 400 | N/A        | 99.92% | 99.986%                |
| 500 | N/A        | N/A    | 99.999%                |

### Runtime 
| $N$ | ITensorMPS | block2 | ITensorMPOConstruction |
|-----|------------|--------|------------------------|
| 10  | 0.32s      | 0.016s | 0.009s                 |
| 20  | 30.6s      | 0.089s | 0.052s                 |
| 30  | 792s       | 0.30s  | 0.14s                  |
| 40  | N/A        | 0.72s  | 0.38s                  |
| 50  | N/A        | 1.5s   | 0.63s                  |
| 100 | N/A        | 20s    | 7.7s                   |
| 200 | N/A        | 489s   | 103s                   |
| 300 | N/A        | 3711s  | 500s                   |
| 400 | N/A        | 18373s | 1554s                  |
| 500 | N/A        | N/A    | 3802s                  |


## Benchmarks: Electronic Structure Hamiltonian

All runtimes below are taken from a single sample on a 2021 MacBook Pro with the M1 Max CPU and 32GB of memory.

### Bond Dimension 
| $N$ | ITensorMPS | block2 | ITensorMPOConstruction |
|-----|------------|--------|------------------------|
| 10  | 227        | 227    | 227                    |
| 20  | 852        | 852    | 852                    |
| 30  | N/A        | 1877   | 1877                   |
| 40  | N/A        | 3302   | 3355                   |
| 50  | N/A        | 5127   | 5134                   |
| 60  | N/A        | 7352   | 7364                   |
| 70  | N/A        | 9977   | 9985                   |
| 80  | N/A        | 13002  | 13006                  |
| 90  | N/A        | 16427  | 16473                  |

### Sparsity 
| $N$ | ITensorMPS | block2 | ITensorMPOConstruction | After `splitblocks` |
|-----|------------|--------|------------------------|---------------------|
| 10  | 94.5%      | 89.2%  | 93.4%                  | 95.7%               |
| 20  | 95.4%      | 93.1%  | 94.1%                  | 97.1%               |
| 30  | N/A        | 95.0%  | 94.4%                  | 97.6%               |
| 40  | N/A        | 96.0%  | 94.4%                  | 97.8%               |
| 50  | N/A        | 96.7%  | 94.6%                  | 97.9%               |
| 60  | N/A        | 97.2%  | 94.6%                  | 97.9%               |
| 70  | N/A        | 97.6%  | 94.7%                  | N/A                 |
| 80  | N/A        | 97.8%  | 94.7%                  | N/A                 |
| 90  | N/A        | 98.1%  | 94.7%                  | N/A                 |

### Runtime 
| $N$ | ITensorMPS | block2 | ITensorMPOConstruction |
|-----|------------|--------|------------------------|
| 10  | 3.65s      | 0.166s | 0.052s                 |
| 20  | 605s       | 2.67s  | 1.21s                  |
| 30  | N/A        | 15.0s  | 7.00s                  |
| 40  | N/A        | 50.4s  | 29.5s                  |
| 50  | N/A        | 137s   | 104s                   |
| 60  | N/A        | 332s   | 284s                   |
| 70  | N/A        | 619s   | 625s                   |
| 80  | N/A        | 1220s  | 1545s                  |
| 90  | N/A        | 1944s  | 3968s                  |