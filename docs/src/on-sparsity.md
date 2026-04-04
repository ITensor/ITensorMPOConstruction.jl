## A note on sparsity

<!-- TODO: Rewrite -->

### Thanks to [Huanchen Zhai](https://scholar.google.com/citations?user=HM_YBL0AAAAJ&hl=en) for providing the discussion and data motivating this section.

The core component of many MPO construction algorithms is to take an operator ``\hat{O}`` defined on a bipartite system

```\math
  \hat{O} = \sum_{a = 1}^{N_A} \sum_{b = 1}^{N_B} \gamma_{ab} \hat{A}_a \otimes \hat{B}_b \ ,
```

and turn it into a two site MPO

```\math
  \hat{O} = \sum_{\chi = 1}^w \left( \sum_{a = 1}^{N_A} \alpha_{\chi a} \hat{A}_a \right) \otimes \left( \sum_{b = 1}^{N_B} \beta_{\chi b} \hat{B}_b \right) \ .
```

The MPO bond dimension is ``w`` and the MPO tensors are essentially operator valued vectors whose ``\chi``-th entry is ``\left( \sum_{a = 1}^{N_A} \alpha_{\chi a} \hat{A}_a \right)`` for the left tensor. This vector of operators can be reshaped into the standard matrix of operators if ``a`` is a combined incoming link and onsite index. 

In the case of operators with a global ``U(1)`` symmetry the matrix ``\gamma`` can be permuted into a block diagonal form. This form has many benefits since each block can be decomposed into a MPO independently and the ``\alpha`` and ``\beta`` matrices inherit the block diagonal nature. The way that ``\gamma`` is brought into block diagonal form in the original `ITensorMPS` algorithm is to use the quantum numbers associated with the operators ``\hat{A}_a`` and ``\hat{B}_b``. This has two problems; first it requires that the user provide the symmetry information, and second there may be other block diagonal blocks unrelated to the symmetries. The approach taken in this library addresses both these issues.

In `ITensorMPOConstruction` ``\gamma`` is brought into block diagonal form by viewing it as a bipartite graph adjacency matrix (``\gamma_{a b} \neq 0`` implies there is an edge between left-vertex ``a`` and right-vertex ``b``) and finding the connected components. Each connected component is then a block in the block diagonal representation. This does not require the use of any symmetry information and is guaranteed to produce the maximum possible number of blocks. However, although `ITensorMPOConstruction` will produce a matrix ``\alpha`` of minimal bond dimension ``w`` and greatest number of diagonal blocks this does not mean that the overall sparsity of the MPO tensor is maximized. This is because `ITensors`' sparse format is more flexible, and most of the time each diagonal block in the ``\alpha`` matrix winds up being stored itself in a sparse format. We use the sparse QR decomposition to decompose each block of ``\gamma``, and while the resulting matrices (which become the blocks in ``\alpha`` and ``\beta``) are sparse, their sparsity is not necessarily optimal.

To illustrate the suboptimal sparsity, we turn to [Block2](https://github.com/block-hczhai/block2-preview) which has a sophisticated set of MPO construction algorithms. Specifically we will use the `FastBipartite` algorithm, based on the bipartite graph algorithm from [RenLi2020](https://doi.org/10.1063/5.0018149). The bipartite algorithm is very efficient and also produces MPO tensors of exceptional sparsity. The drawback is that it is unable to compress the MPO bond dimension. For example, for the momentum space Fermi-Hubbard Hamiltonian the bond dimension it produces is ``O(N^2)``. However, for some operators such as the electronic structure Hamiltonian the bipartite algorithm and the rank decomposition algorithm (used in `ITensorMPS` and here) produce similar MPO bond dimensions. In these cases, the bipartite algorithm will likely produce MPOs of greater sparsity.

In the table below we present data from constructing two different electronic structure Hamiltonians, the second of which is from [ZhaiLee2023](https://doi.org/10.1021/acs.jpca.3c06142). Our rank decomposition algorithm only slightly reduces the bond dimension compared to the bipartite MPO from `Block2`, but it results in a much denser MPO. This increase in sparsity has a significant impact on the subsequent DMRG performance, which is larger by 75% for our rank decomposition MPO (timings and sparsities taken from `Block2`'s by transferring over the MPO from `ITensorMPOConstruction`).

| system                                                                              | algorithm | bond dimension | sparsity |
|-------------------------------------------------------------------------------------|-----------|----------------|----------|
| C <sub>2</sub>                                                                      | rank      | 704            | 97.21%   |
| C <sub>2</sub>                                                                      | bipartite | 722            | 98.68%   |
| [Fe <sub>2</sub> S(C H <sub>3</sub>) (S C H <sub>3</sub>)<sub>4</sub>]<sup>3-</sup> | rank      | 2698           | 88.70%   |
| [Fe <sub>2</sub> S(C H <sub>3</sub>) (S C H <sub>3</sub>)<sub>4</sub>]<sup>3-</sup> | bipartite | 2738           | 95.64%   |

To further complicate matters `Block2` has a different, less flexible, sparse storage format from `ITensors`. Specifically, they store MPO tensors in a "matrix of operators" format, where the onsite operator is always dense. Essentially this format stores a sparse representation of ``\alpha``, while maintaining a dense form for ``\hat{A}_a``. To facilitate comparisons between the two libraries without having to convert the MPOs between them, we provide the function `block2_nnz(mpo::MPO)::Tuple{Int, Int}` which returns the total number of blocks (the size of ``\alpha`` summed across each site) and the number of non-zero blocks.