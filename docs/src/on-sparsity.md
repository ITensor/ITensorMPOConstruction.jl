## A note on sparsity

### Thanks to [Huanchen Zhai](https://scholar.google.com/citations?user=HM_YBL0AAAAJ&hl=en) for providing the discussion and data motivating this section.

The core component of many MPO construction algorithms is to take an operator ``\hat{O}`` defined on a bipartite system

````math
  \hat{O} = \sum_{a = 1}^{N_A} \sum_{b = 1}^{N_B} \gamma_{ab} \hat{A}_a \otimes \hat{B}_b \ ,
````

and turn it into a two site MPO

````math
  \hat{O} = \sum_{\chi = 1}^w \left( \sum_{a = 1}^{N_A} \alpha_{\chi a} \hat{A}_a \right) \otimes \left( \sum_{b = 1}^{N_B} \beta_{\chi b} \hat{B}_b \right) \ .
````

The MPO bond dimension is ``w`` and the MPO tensors are essentially operator valued vectors whose ``\chi``-th entry is ``\left( \sum_{a = 1}^{N_A} \alpha_{\chi a} \hat{A}_a \right)`` for the left tensor. This vector of operators can be reshaped into the standard matrix of operators if ``a`` is a combined incoming link and onsite index. 

In the case of operators with a global ``U(1)`` symmetry the matrix ``\gamma`` can be permuted into a block diagonal form. This form has many benefits since each block can be decomposed independently and the ``\alpha`` and ``\beta`` matrices inherit the block diagonal nature.

In ITensorMPOConstruction ``\gamma`` is brought into block diagonal form by viewing it as a bipartite graph adjacency matrix (``\gamma_{a b} \neq 0`` implies there is an edge between left-vertex ``a`` and right-vertex ``b``) and finding the connected components. Each connected component is then a block in the block diagonal representation. This does not require the use of any symmetry information and is guaranteed to produce the maximum possible number of blocks. However, although ITensorMPOConstruction will produce a matrix ``\alpha`` of minimal bond dimension ``w`` and greatest number of diagonal blocks *this does not mean that the overall sparsity of the MPO tensor is maximized*. In certain important cases, such as the electronic structure Hamiltonian, an MPO with a slightly larger bond dimension can be much sparser, and in turn lead to improved DMRG performance.

To illustrate this situation we turn to the `FastBipartite` algorithm from [Block2](https://github.com/block-hczhai/block2-preview) based on the bipartite graph MPO construction algorithm from [RenLi2020](https://doi.org/10.1063/5.0018149). The bipartite algorithm is very efficient and also produces MPO tensors of exceptional sparsity. The drawback is that it is unable to compress the MPO bond dimension. For example, for the momentum space Fermi-Hubbard Hamiltonian the bond dimension it produces is ``O(N^2)`` as compared to the optimal ``O(N)``. However, for some operators such as the electronic structure Hamiltonian the bipartite algorithm and the rank decomposition algorithm (used in ITensorMPS and here) produce similar MPO bond dimensions. In these cases, the bipartite algorithm will likely produce MPOs of greater sparsity.

In the table below we present data from constructing two different electronic structure Hamiltonians, the second of which is from [ZhaiLee2023](https://doi.org/10.1021/acs.jpca.3c06142). Our rank decomposition algorithm only slightly reduces the bond dimension compared to the bipartite MPO from `Block2`, but it results in a much denser MPO. This increase in density has a significant impact on the subsequent DMRG performance, which takes 75% longer for our rank decomposition MPO (timings and sparsities taken from `Block2` after transferring over the MPO from ITensorMPOConstruction).

| system                                                                     | algorithm | bond dimension | sparsity |
|----------------------------------------------------------------------------|-----------|----------------|----------|
| ``\text{C}_2``                                                             | rank      | 704            | 97.21%   |
| ``\text{C}_2``                                                             | bipartite | 722            | 98.68%   |
| ``\left[\text{Fe}_2 \text{S} (\text{C H}_3)(\text{SCH}_3)_4 \right]^{3-}`` | rank      | 2698           | 88.70%   |
| ``\left[\text{Fe}_2 \text{S} (\text{C H}_3)(\text{SCH}_3)_4 \right]^{3-}`` | bipartite | 2738           | 95.64%   |

To further complicate matters `Block2` has a different, less flexible, sparse storage format from ITensors. Specifically, they store MPO tensors in a "matrix of operators" format, where the onsite operator is always dense. Essentially this format stores a sparse representation of ``\alpha``, while maintaining a dense form for ``\hat{A}_a``. To facilitate comparisons between the two libraries without having to convert the MPOs between them, we provide the function `block2_nnz(mpo::MPO)::Tuple{Int, Int}` which returns the total number of blocks (the size of ``\alpha`` summed across each site) and the number of non-zero blocks.
