# Symbolic Vertex-Cover MPO Construction

## Summary

Add a reusable symbolic MPO construction path for integer-labeled `OpIDSum`s.
`MPO_symbolic` should run the vertex-cover MPO construction once, store the
resulting tensor entries as integer linear combinations of coefficient
variables, and allow repeated numeric instantiation with different coefficient
vectors.

The primary API should follow the package's current argument order:

```julia
sym = MPO_symbolic(os, sites; basis_op_cache_vec=nothing,
                   check_for_errors=true,
                   combine_qn_sectors=false,
                   output_level=0)

H = instantiate_MPO(sym, coefficients; splitblocks=true, checkflux=false)
instantiate_MPO!(H, sym, coefficients; checkflux=false)
H2 = instantiate_MPO(H, sym, coefficients; checkflux=false)
```

## Public API and V1 Semantics

- Export `SymbolicMPO` and `MPO_symbolic`.
- Support only `OpIDSum{N,<:Integer,Ti}` input in V1.
- Symbolic construction is vertex-cover construction. It should not expose an
  `alg` keyword, and there is no QR symbolic path to support.
- Do not add symbolic `OpSum` conversion in V1.
- Preserve numeric `MPO_new` behavior exactly.
- User-provided integer labels are variable labels and must always be greater
  than zero.
- Convert each user label `k` to internal symbolic id `k + 1`, reserving
  internal id `1` for the constant one introduced by the vertex-cover algorithm.
- Carry signs by negating the internal symbolic id. Negative ids can arise from
  sign-only basis rewrites and other internal sign handling; they are not
  user-facing labels.
- During numeric substitution:
  - internal id `+1` evaluates to `+1.0`,
  - internal id `-1` evaluates to `-1.0`,
  - internal id `k > 1` evaluates to `coefficients[k - 1]`,
  - internal id `k < -1` evaluates to `-coefficients[-k - 1]`.
- The coefficient vector is indexed by the original positive user labels. The
  reserved internal constant id is transparent to the user, so there is no
  reserved slot in `coefficients`.
- Reject `OpSum` input, unsupported basis rewrite factors, and missing
  coefficients with clear `ArgumentError`s.

## Internal Representation

### Symbolic local matrix terms

Do not model symbolic MPO entries as scalar coefficients. A symbolic block entry
is a symbolic weighted sum of local operator matrices:

```julia
const SymbolicLocalMatrix{Ti<:Integer} = Vector{Tuple{Int,Ti}}
```

Each tuple is `(signed_weight_id, signed_local_op_id)`.

- `signed_weight_id` is the signed internal coefficient id:
  - `+1` and `-1` are the constants `+1` and `-1`,
  - `k > 1` maps to user assignment `coefficients[k - 1]`,
  - `k < -1` maps to `-coefficients[-k - 1]`.
- `abs(signed_local_op_id)::Ti` uses the same integer type as
  `OpIDSum{N,C,Ti}` and identifies the local operator being scaled by that
  signed coefficient id.
- `signed_local_op_id < 0` means the local operator should be emitted with the
  Jordan-Wigner string applied. `signed_local_op_id > 0` means no
  Jordan-Wigner string is applied.
- A term represents
  `substitute(signed_weight_id, coefficients) * local_matrix`, where
  `local_matrix` is `op_cache_vec[n][abs(signed_local_op_id)].matrix`, with the
  Jordan-Wigner sign pattern applied when `signed_local_op_id < 0`.

Required helpers:

- `_weight_value(signed_weight_id, coefficients)` substitutes one signed
  internal id using the rules above.
- `_normalize_symbolic_local_matrix!(terms)` canonicalizes ordering and removes
  canceling pairs such as `(k, op)` with `(-k, op)`. It must not collapse
  repeated identical pairs into one entry, since multiplicity represents
  repeated contributions such as `2 * coefficients[k - 1] * op`.
- `_scale_symbolic_weight(signed_weight_id, factor)` accepts only exact factors
  `+1` and `-1` in V1, flipping the signed id for `-1`.
- `_evaluate_symbolic_local_matrix(terms, coefficients, op_cache)` builds the
  dense numeric local matrix by summing cached local operator matrices, applying
  Jordan-Wigner strings for negative local operator ids, and substituting
  weights.

### Symbolic Block Storage

Do not reuse `BlockSparseMatrix{C}` for symbolic block entries, since that alias
stores dense `Matrix{C}` blocks. Symbolic construction should use:

```julia
const SymbolicBlockSparseMatrix{Ti<:Integer} =
  Vector{Dictionary{Int,SymbolicLocalMatrix{Ti}}}
```

This mirrors the existing `right_link => left_link => block` layout, but each
block is a symbolic local matrix rather than an already-expanded dense matrix.
Numeric instantiation converts these symbolic blocks into ordinary
`BlockSparseMatrix{T}` values immediately before constructing `ITensor`s.

## Implementation Checklist

### 1. Add and test symbolic local matrix operations

- Add `SymbolicLocalMatrix{Ti}` and `SymbolicBlockSparseMatrix{Ti}` aliases or
  small wrapper structs.
- Implement helpers for appending terms, normalizing terms, applying `+1` or
  `-1` rewrite factors, computing `max_user_label`, and substituting signed
  internal ids.
- Implement `_evaluate_symbolic_local_matrix` to produce a dense numeric matrix
  for one site from `(signed_weight_id, signed_local_op_id)` terms and the site's
  operator cache.
- Add focused tests for:
  - duplicate `(weight_id, signed_local_op_id)` terms preserving multiplicity,
  - zero normalization,
  - `+1` and `-1` constant substitution,
  - positive user label to internal id mapping,
  - signed internal id evaluation,
  - missing coefficient rejection,
  - non-`±1` rewrite-factor rejection.

Tests for this step:

- Unit-test `_weight_value` directly:
  - `+1` returns `1`,
  - `-1` returns `-1`,
  - `+3` returns `coefficients[2]`,
  - `-3` returns `-coefficients[2]`,
  - `+4` throws when `length(coefficients) == 2`.
- Unit-test symbolic local matrix normalization:
  - `[(3, 2), (-3, 2)]` normalizes to an empty term list,
  - `[(3, 2), (3, 2)]` keeps two entries and evaluates to twice the same local
    matrix contribution,
  - terms are sorted deterministically so equality checks are stable.
- Unit-test Jordan-Wigner encoding:
  - `(3, -2)` evaluates to `coefficients[2]` times operator `2` with the
    Jordan-Wigner sign pattern,
  - `(3, 2)` evaluates to the same operator without the Jordan-Wigner sign
    pattern.

### 2. Convert the integer-labeled `OpIDSum` into internal-id form

- Add an internal conversion helper:

```julia
internalize_symbolic_ids!(os::OpIDSum{N,C,Ti}) where {N,C<:Integer,Ti}
```

- This helper may mutate the provided `OpIDSum`, matching the existing MPO
  construction behavior where preprocessing, sorting, and duplicate compaction
  are destructive.
- Map each stored positive scalar label to an internal id in place.
- Reject scalar labels less than or equal to `0`.
- Map stored scalar label `k > 0` to internal id `k + 1`.
- Keep the same `op_cache_vec`, `abs_tol`, and `modify!`.
- Add tests proving the in-place conversion maps labels correctly and rejects
  nonpositive labels.

Tests for this step:

- Build a small `OpIDSum{2,Int,Int}` with scalar labels `1`, `2`, and `3`;
  after `internalize_symbolic_ids!`, assert that the stored labels are `2`, `3`,
  and `4`.
- Add terms with labels `0` and `-1` and assert `internalize_symbolic_ids!`
  throws an `ArgumentError`.
- Assert the helper is intentionally mutating by checking that the same `os`
  object has updated scalar storage after conversion.
- Assert `op_cache_vec`, `abs_tol`, and `modify!` are still the original objects
  or equivalent values after conversion.

### 3. Make basis rewrite compatible with internal signed ids

- Reuse `prepare_opID_sum!(os, basis_op_cache_vec; symbolic_coefficients=true)`
  after the in-place internal-id conversion. The keyword keeps ordinary numeric
  `MPO_new` preprocessing on the existing coefficient-scaling path while the
  symbolic path treats integer coefficients as internal signed ids.
- Ensure `rewrite_in_operator_basis!` can multiply internal signed ids by basis
  rewrite factors only when the factor is exactly `+1` or `-1`.
- Reject `im`, `2im`, `0.5`, `2`, and other unsupported factors with an
  `ArgumentError`; this may be reported through Julia's threaded task failure
  wrapper because symbolic rewrites use the same threaded loop as numeric
  rewrites.
- Add a sign-only rewrite regression where a same-site product produces `-1`
  and verifies the internal id sign flips while the user label is preserved.

Tests for this step:

- Construct a small operator-cache fixture where a same-site product rewrites
  into a basis operator with factor `-1`; after preparation, verify internal id
  `k + 1` becomes `-(k + 1)` and the basis operator id is updated.
- Construct a fixture with rewrite factor `+1`; verify the internal id sign is
  unchanged.
- Reuse the existing `X * Y -> im * Z` style case or an equivalent fixture to
  assert complex factors throw.
- Add separate fixtures for `0.5` and `2` factors and assert both throw, since
  V1 accepts only exact `±1`.

### 4. Build a symbolic graph without adding identifier integers

- Numeric `MPOGraph(os::OpIDSum)` must keep its current duplicate-term
  compaction and tolerance behavior.
- Add a symbolic graph-construction path for the internal-id `OpIDSum`.
- Duplicate operator tuples must combine correctly even when their scalar labels
  are different symbolic variables. Combining duplicates should preserve all
  signed ids, for example by storing a vector of signed internal ids as the edge
  weight or by expanding the duplicates into distinct edge entries before local
  block construction.
- The graph edge weight type should remain a signed internal id or a collection
  of signed ids, not a symbolic local matrix term.
- Add a test where duplicate operator tuples with different user labels both
  contribute to the same symbolic block and instantiate correctly.

Tests for this step:

- Build an `OpIDSum{2,Int,Int}` with two identical operator tuples labeled `1`
  and `2`; after symbolic graph construction and duplicate-right-vertex
  compaction, assert the resulting graph preserves both signed ids as separate
  edge weights, not their numeric sum.
- Add two identical operator tuples with the same label `1`; instantiate with
  `coefficients[1] = a` through the symbolic local matrix helper and verify the
  local contribution is `2a`.
- Add two identical operator tuples whose labels become opposite signed internal
  ids after preprocessing; verify the derived symbolic local matrix cancels or
  evaluates to zero for that contribution.

### 5. Build `SymbolicMPO`

Add:

```julia
struct SymbolicMPO{Ti<:Integer}
  offsets::Vector{Vector{Int}}
  block_sparse_matrices::Vector{Vector{SymbolicBlockSparseMatrix{Ti}}}
  sites::Vector{<:Index}
  llinks::Vector{Index}
  op_cache_vec::OpCacheVec
  max_user_label::Int
end
```

If Julia requires concrete field types for `sites`, parameterize the site vector
type instead of using an abstract field.

`MPO_symbolic(os, sites; kwargs...)` should:

- validate `os` has an integer coefficient type,
- reject any supplied `alg` keyword with an `ArgumentError` explaining that
  symbolic construction is always vertex-cover based,
- convert `basis_op_cache_vec` with `to_OpCacheVec(sites, basis_op_cache_vec)`,
- convert `os` to internal-id form in place,
- run `prepare_opID_sum!` and optionally `check_os_for_errors`,
- build the symbolic graph without numerically adding identifier ids,
- initialize the dummy left link exactly as `MPO_new` does,
- call the existing `resume_MPO_construction!` with `alg="VC"` and symbolic
  block storage, so symbolic construction reuses `at_site!` and
  `process_vertex_cover!` instead of maintaining parallel symbolic driver
  functions,
- use small storage-dispatch helpers inside the vertex-cover path so numeric
  blocks still accumulate dense matrices while symbolic blocks append
  `(signed_weight_id, signed_local_op_id)` terms,
- when adding a local contribution, append
  `(signed_weight_id, signed_local_op_id)` to the symbolic block instead of
  expanding the dense local operator matrix,
- store offsets, symbolic block matrices, sites, links, `op_cache_vec`, and the
  maximum positive user label needed for substitution.

Tests for this step:

- Construct a small transverse-field Ising `OpIDSum{2,Int,Int}` and assert
  `MPO_symbolic(os, sites)` returns a `SymbolicMPO` with:
  - `length(offsets) == length(sites)`,
  - `length(block_sparse_matrices) == length(sites)`,
  - `length(llinks) == length(sites) + 1`,
  - `max_user_label` equal to the largest user label in the input.
- Assert `MPO_symbolic(os, sites; alg="QR")` throws an `ArgumentError`.
- Assert `MPO_symbolic(::OpSum, sites)` throws an `ArgumentError` or has no
  method, whichever API choice is implemented.
- For a fermionic fixture, inspect at least one symbolic local matrix term and
  assert negative `signed_local_op_id` is used where `needs_JW_string` is true.

### 6. Instantiate a fresh numeric MPO

Add:

```julia
instantiate_MPO(sym::SymbolicMPO, coefficients; splitblocks=true, checkflux=false)
```

It should:

- validate the coefficient vector length against `sym.max_user_label`,
- evaluate symbolic block entries into numeric block matrices using
  `sym.op_cache_vec`,
- preserve the symbolic structural pattern even when evaluated values are zero,
- call the existing tensor assembly path with the requested `splitblocks` value,
- contract away the dummy boundary links the same way as the low-level
  `instantiate_MPO(offsets, block_sparse_matrices, sites, llinks; ...)`,
- optionally run `ITensors.checkflux`.

For QN tensors, structural block discovery must be based on symbolic local
matrix terms before numeric substitution, so a coefficient vector containing
zeros does not change the block structure or link layout.

Tests for this step:

- Instantiate a small symbolic MPO with two different coefficient vectors and
  compare each result against numeric `MPO_new` built from an equivalent numeric
  `OpIDSum`.
- Run the duplicate-label symbolic graph regression through `MPO_symbolic` and
  compare the instantiated MPO against numeric `MPO_new` built from the
  corresponding numeric coefficients.
- Instantiate with `splitblocks=false` and `splitblocks=true` for a QN-conserving
  fixture and compare both against numeric `MPO_new`.
- Instantiate with a coefficient vector containing zeros and assert the link
  dimensions match a nonzero-coefficient instantiation from the same
  `SymbolicMPO`.
- Pass a too-short coefficient vector and assert an `ArgumentError`.
- Run with `checkflux=true` on a QN fixture and assert it succeeds.

### 7. Add in-place and template-assisted instantiation

Add:

```julia
instantiate_MPO!(H::MPO, sym::SymbolicMPO, coefficients; checkflux=false)
instantiate_MPO(H_template::MPO, sym::SymbolicMPO, coefficients; checkflux=false)
```

These overloads must not accept a `splitblocks` keyword. A preconstructed MPO
already fixes the block layout, so evaluation must use the layout of `H` or
`H_template`.

`instantiate_MPO!` should:

- check `length(H) == length(sym.sites)`,
- check site indices are compatible with `sym.sites`,
- check link dimensions match `sym.llinks`,
- overwrite each tensor in `H` with the evaluated tensor for that site,
- preserve the existing `MPO` object identity.

`instantiate_MPO(H_template, sym, coefficients)` should:

- reject incompatible templates with `ArgumentError`,
- copy or reuse a compatible template,
- call `instantiate_MPO!`,
- return the instantiated MPO.

It is acceptable for V1 to overwrite tensors rather than doing lower-level
storage mutation, as long as the public behavior and link structure are stable.

Tests for this step:

- Build `H = instantiate_MPO(sym, coeffs1; splitblocks=true)`, then call
  `instantiate_MPO!(H, sym, coeffs2)` and compare `H` against
  `instantiate_MPO(sym, coeffs2; splitblocks=true)`.
- Assert `instantiate_MPO!(H, sym, coeffs; splitblocks=false)` throws a
  `MethodError` or `ArgumentError`, since this overload must not accept
  `splitblocks`.
- Call `instantiate_MPO(H, sym, coeffs2)` with a compatible template and compare
  against fresh instantiation using the same layout.
- Build an incompatible template with different sites or link dimensions and
  assert `instantiate_MPO(H_bad, sym, coeffs)` throws an `ArgumentError`.

### 8. Add symbolic MPO tests

Create `test/symbolic-mpo-tests.jl` and include it from `test/runtests.jl`.

The file should use named `@testset`s matching the implementation steps, so a
failure points directly to the layer that broke:

- `"symbolic local matrix terms"`:
  - normalization,
  - duplicate `(weight_id, signed_local_op_id)` multiplicity,
  - constant weight ids,
  - negative local operator id applies the Jordan-Wigner string,
  - positive user label to internal id mapping,
  - signed internal ids,
  - missing coefficient error,
  - unsupported rewrite-factor error.
- `"internal symbolic ids"`:
  - in-place user-label to internal-id conversion,
  - nonpositive label rejection,
  - negative internal ids from later sign-carrying operations remain supported.
- `"basis rewrite"`:
  - `+1` rewrite factor preserves id sign,
  - `-1` rewrite factor flips id sign,
  - complex, fractional, and integer factors other than `±1` throw.
- `"symbolic graph duplicate terms"`:
  - different labels on the same operator tuple are both preserved,
  - repeated identical labels preserve multiplicity,
  - opposite signed ids cancel or instantiate to zero.
- `"fresh instantiation"`:
  - build `OpIDSum{2,Int,Int}`,
  - construct `sym = MPO_symbolic(os, sites)`,
  - instantiate with several random coefficient vectors indexed directly by the
    positive user labels, passing `splitblocks` to fresh `instantiate_MPO`,
  - compare against numeric `MPO_new` built with the same coefficients.
- `"public API rejections"`:
  - any supplied `alg` keyword,
  - `OpSum` input,
  - missing coefficient,
  - incompatible preconstructed MPO templates.
- `"in-place instantiation"`:
  - instantiate once,
  - update with a second coefficient vector through `instantiate_MPO!`,
  - verify `instantiate_MPO!` does not accept a `splitblocks` keyword,
  - compare to fresh instantiation,
  - check link dimensions remain unchanged.
- `"template-assisted instantiation"`:
  - instantiate from a compatible template,
  - compare to fresh instantiation,
  - reject an incompatible template.

Tests for this step:

- `julia --project test/symbolic-mpo-tests.jl` runs all new symbolic testsets
  by itself.
- `julia --project test/runtests.jl` includes `test/symbolic-mpo-tests.jl` once
  and still runs the existing test files.
- Existing `test/ops-tests.jl` still passes, proving the basis-rewrite changes
  did not regress the numeric `OpIDSum` path.

### 9. Update electronic-structure example

Add a compact symbolic section to `examples/electronic-structure.jl`.

Implementation outline:

- Build a small integer-labeled `OpIDSum` in the same term order as the numeric
  electronic-structure `OpIDSum`.
- Keep a `coefficients` vector indexed directly by positive user labels.
- Whenever user label `k` is added to the symbolic `OpIDSum`, store the actual
  numeric coefficient at `coefficients[k]`.
- Construct `sym = MPO_symbolic(
  os_symbolic, sites; basis_op_cache_vec=os_symbolic.op_cache_vec, check_for_errors=false
  )`.
- Instantiate twice with two coefficient vectors, passing `splitblocks=true` to
  fresh `instantiate_MPO`.
- For a small `N`, compare one symbolic instantiation to the existing numeric
  `MPO_new` result.
- Keep the example section short so the existing benchmark narrative remains
  readable.

Tests for this step:

- Run the example with a small `N` setting first and assert the symbolic
  instantiation matches the numeric `MPO_new` result using the existing MPO
  comparison helper or an equivalent relative-difference check.
- Run the final example command:

```bash
julia --project -t8 examples/electronic-structure.jl
```

- Confirm the example still reports the existing numeric construction metrics
  and additionally prints or checks the symbolic construction section.
- Confirm changing the second coefficient vector changes the instantiated MPO
  without rebuilding the symbolic structure.

### 10. Update docs

- Add docstrings for `SymbolicMPO`, `MPO_symbolic`, and the new
  `instantiate_MPO` / `instantiate_MPO!` overloads.
- Update `docs/src/documentation/MPO_new.md` to include the symbolic API in the
  `@docs` block.
- Add a short prose section explaining:
  - integer labels in `OpIDSum`,
  - positive user labels and transparent internal remapping,
  - repeated instantiation,
  - why symbolic construction is vertex-cover based and has no QR option,
  - `±1` rewrite-factor limitation,
  - `OpIDSum`-only limitation.

Tests for this step:

- Verify `julia --project -e 'using ITensorMPOConstruction'` loads cleanly after
  adding exports and docstrings.
- Verify the docs build:

```bash
julia --project=docs --color=yes docs/make.jl
```

- Check the generated docs page includes `SymbolicMPO`, `MPO_symbolic`,
  `instantiate_MPO(sym, ...)`, and `instantiate_MPO!(H, sym, ...)`.

## Verification Checklist

Run focused checks first:

```bash
julia --project -e 'using ITensorMPOConstruction'
julia --project test/symbolic-mpo-tests.jl
julia --project test/ops-tests.jl
julia --project test/test-MPOConstruction.jl
```

Then run broader checks:

```bash
julia --project test/runtests.jl
julia --project=docs --color=yes docs/make.jl
```

Because `examples/electronic-structure.jl` is modified, also run:

```bash
julia --project -t8 examples/electronic-structure.jl
```

## Acceptance Criteria

- Existing numeric `MPO_new` tests pass unchanged.
- `MPO_symbolic` may mutate its input `OpIDSum`, consistently with existing MPO
  construction preprocessing.
- Repeated symbolic instantiation matches fresh numeric MPO construction.
- In-place symbolic instantiation updates an existing MPO without changing link
  dimensions.
- Symbolic construction rejects unsupported input cases with clear errors.
- Docs build successfully and include the new public API.
