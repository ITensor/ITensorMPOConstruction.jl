# Which algorithm is better?

The answer is, of course, system-dependent, and we recommend trying both to see which performs better for your specific system. Below we list the pros and cons of each algorithm, in case you want to build an intuition for why one works better than the other.

## The QR decomposition (`alg = "QR"`)

Best for systems where the coefficients of the terms in the `OpSum` have structure. The [Momentum Space Fermi-Hubbard](examples/fermi-hubbard-ks.md) Hamiltonian is a prime example of such a system.

### Pros
* Produces the minimum bond dimension in all cases.
* Sparsity is optimal with `splitblocks = false`.

### Cons
* With `splitblocks = true` the sparsity can be poor.
* Uses more memory and is slower than the vertex cover algorithm for unstructured problems.

## The minimum vertex cover algorithm (`alg = "VC"`)

Best for systems where the coefficients of the terms in the `OpSum` are unstructured or essentially random. The [Electronic Structure](examples/electronic-structure.md) Hamiltonian is one such system.

### Pros
* Produces MPOs of optimal or near-optimal bond dimension for unstructured problems.
* Very high sparsity with `splitblocks = true`.
* Fast with little memory overhead.

### Cons
* When the coefficients have structure, the resulting bond dimension can be far from optimal. In the worst case it can produce an MPO of exponential bond dimension when the optimal bond dimension is one.

# Algorithmic details

Here we go into detail on the differences between the two algorithms and define what it means for a problem to be "structured".

The core of both the QR decomposition and vertex cover algorithms is to take an operator ``H`` defined on a bipartite system

```math
  H = \sum_{a = 1}^{N_A} \sum_{b = 1}^{N_B} \gamma_{ab} A_a \otimes B_b \ ,
```

and turn it into a two-site MPO

```math
  H = \sum_{\chi = 1}^m \left( \sum_{a = 1}^{N_A} \alpha_{\chi a} A_a \otimes \sum_{b = 1}^{N_B} \beta_{\chi b} B_b \right) \ .
```

The MPO bond dimension is ``m`` and the MPO tensors are essentially operator-valued vectors whose ``\chi``-th entry is ``\left( \sum_{a = 1}^{N_A} \alpha_{\chi a} \hat{A}_a \right)`` for the left tensor. This vector of operators can be reshaped into the standard matrix of operators if ``a`` is interpreted as a combined incoming-link and onsite-index label. Assuming that ``\{ A \}_{a = 1}^{N_A}`` and ``\{ B \}_{b = 1}^{N_B}`` are each linearly independent, the condition becomes

```math
\gamma_{ab} = \sum_{\chi = 1}^m \alpha_{\chi a} \beta_{\chi b} \rightarrow \gamma = \alpha^T \beta\ .
```

The minimum bond dimension, meaning the minimum value of ``m``, is exactly the rank of the matrix ``\gamma``, and the matrices ``\alpha`` and ``\beta`` can be obtained from any rank decomposition of ``\gamma``. In the QR algorithm, we use the sparse rank-revealing QR decomposition from [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).

The minimum vertex cover algorithm, however, takes a different approach, which is why it is not able to achieve the minimum bond dimension in all cases. Instead of performing a rank decomposition, it views the bipartite operator as defining an unweighted bipartite graph, with an edge connecting vertex ``a`` on the left with vertex ``b`` on the right if ``\gamma_{ab} \neq 0``. It then constructs a minimum vertex cover of this graph, the smallest set of vertices such that every edge has at least one endpoint in the set. Given the set of left vertices in the cover, ``C_A``, and the set of right vertices in the cover, ``C_B``, the resulting MPO is

```math
H = \sum_{a \in C_A} \left( A_a \otimes \sum_{b = 1}^{N_B} \gamma_{a b} B_b \right) + \sum_{b \in C_B} \left( \sum_{a = 1}^{N_A} \gamma_{a b} A_{a} \otimes B_b \right)
```

where for simplicity we assumed no edge was covered twice. Now the bond dimension of the MPO is ``m = |C_A| + |C_B|``. This bond dimension turns out to be the [structural rank](https://www.mathworks.com/help/matlab/ref/sprank.html) of ``\gamma``, the maximum rank among all matrices with the same nonzero pattern.

So, if ``\operatorname{rank}(\gamma) \ll \operatorname{structural\,rank}(\gamma)``, then the problem is structured and the QR decomposition algorithm will be by far the superior option. If instead, ``\operatorname{rank}(\gamma) \approx \operatorname{structural\,rank}(\gamma)``, then the problem is unstructured and the minimum vertex cover algorithm will perform well.
