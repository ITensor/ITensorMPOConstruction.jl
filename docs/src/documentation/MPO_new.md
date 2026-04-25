# MPO_new
The main exported function is `MPO_new` which takes an `OpSum` and transforms it into a MPO. This algorithm has three constraints:

1. The operator must be expressed as a sum of products of single site operators. For example a CNOT could not appear in the sum since it is a two site operator.

2. When dealing with Fermionic systems the parity of each term in the sum must be even. That is the combined number of creation and annihilation operators in each term in the sum must be even. It should be possible to relax this constraint.

3. Each term in the sum of products representation can only have a single operator acting on a site. For example a term such as $\mathbf{X}^{(1)} \mathbf{X}^{(1)}$ is not allowed. However, there is a pre-processing utility that can automatically replace $\mathbf{X}^{(1)} \mathbf{X}^{(1)}$ with $\mathbf{I}^{(1)}$.

```@docs
MPO_new
resume_MPO_construction!
instantiate_MPO
sparsity
block2_nnz
```

## A note on truncation:
`MPO_new` is designed to construct _numerically exact_ MPOs, the tolerance parameter should really only be used to adjust the definition of "numerically exact" and not to perform truncation. If truncation is desired, truncate the resulting MPO with `ITensorMPS.truncate`. See [Haldane-Shastry and truncation](../examples/haldane-shastry.md) for an example.
