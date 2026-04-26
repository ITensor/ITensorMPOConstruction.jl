_resume_MPO_kwargs = """
- `combine_qn_sectors=false`: When `true`, the blocks of the MPO tensors corresponding to the same
  quantum numbers are merged together into a single block. This can decrease the resulting sparsity,
  but can offer performance gains.
- `alg="QR"`: The decomposition algorithm to use. The options are `"QR"`, which uses the rank decomposition
  algorithm based on the sparse QR decomposition, and `"VC"`, which uses the vertex cover algorithm. When
  the vertex cover algorithm is used, all tolerance arguments are ignored.
- `tol=1`: A multiplicative modifier to the default tolerance used in SPQR's sparse QR decomposition,
  see [SPQR user guide Section 2.3](https://fossies.org/linux/SuiteSparse/SPQR/Doc/spqr_user_guide.pdf).
  The value of the default tolerance depends on the input matrix, which means a different
  tolerance is used for each decomposition. In the cases we have examined, the default
  tolerance works well for producing accurate MPOs.
- `absolute_tol=false`: If `true`, override the default adaptive tolerance scheme outlined above,
  and use the value of `tol` as the single tolerance for each decomposition.
- `call_back`: A function that is called after constructing the MPO tensor at `cur_site`.
  Called as `call_back(n, offsets, block_sparse_matrices, sites, llinks, g, op_cache_vec)`.
  Primarily used for writing checkpoints to disk for large calculations.
- `output_level=0`: Controls progress and timing output.
"""

@doc """
    resume_MPO_construction!(
      n_init, offsets, block_sparse_matrices, sites, llinks, g, op_cache_vec;
      combine_qn_sectors=false, alg="QR", tol=1, absolute_tol=false,
      call_back=..., output_level=0
    ) -> Nothing

Continue MPO construction starting from site `n_init` using the current graph
state `g`.

This is the low-level driver underlying `MPO_new`. At each site it factors the
current graph with `at_site!`, stores component offsets and intermediate
block-sparse local tensor data in `offsets[n]` and
`block_sparse_matrices[n]`, and advances to the next-site graph. The callback is
invoked after each site is completed, before moving on to the next site.

`offsets` and `block_sparse_matrices` are preallocated vectors with one entry
per site. `llinks` is a length `length(sites) + 1` vector whose first link is
already initialized; this routine fills the outgoing link for each processed
site.

Keyword arguments:
$_resume_MPO_kwargs
"""
function resume_MPO_construction!(
  n_init::Int,
  offsets::Vector{Vector{Int}},
  block_sparse_matrices::Vector{Vector{BlockSparseMatrix{ValType}}},
  sites::Vector{<:Index},
  llinks::Vector{<:Index},
  g::MPOGraph,
  op_cache_vec::OpCacheVec;
  combine_qn_sectors::Bool=false,
  alg::String="QR",
  tol::Real=1,
  absolute_tol::Bool=false,
  call_back::Function=(
    cur_site::Int,
    offsets::Vector{Vector{Int}},
    block_sparse_matrices::Vector{Vector{BlockSparseMatrix{ValType}}},
    sites::Vector{<:Index},
    llinks::Vector{<:Index},
    cur_graph::MPOGraph,
    op_cache_vec::OpCacheVec,
  ) -> nothing,
  output_level::Int=0,
)::Nothing where {ValType<:Number}
  @assert !ITensors.using_auto_fermion() # TODO: This should be fixed.
  @assert tol >= 0

  for n in n_init:length(sites)
    output_level > 1 && println(
      "At site $n/$(length(sites)) the graph takes up $(Base.format_bytes(Base.summarysize(g)))",
    )
    @time_if output_level 1 "at_site!" g, offsets[n], block_sparse_matrices[n], llinks[n + 1] = at_site!(
      ValType,
      g,
      n,
      sites,
      tol,
      absolute_tol,
      op_cache_vec,
      alg;
      combine_qn_sectors,
      output_level,
    )

    call_back(n, offsets, block_sparse_matrices, sites, llinks, g, op_cache_vec)
  end

  return nothing
end

"""
    instantiate_MPO(offsets, block_sparse_matrices, sites, llinks; splitblocks, checkflux) -> MPO

Convert intermediate block-sparse MPO tensor data into an ITensor `MPO`.

For each site `n`, this calls `to_ITensor(offsets[n], block_sparse_matrices[n],
llinks[n], llinks[n + 1], sites[n]; splitblocks)` to assemble the local tensor.
Afterward, the dummy left and right boundary links are contracted away. If
`checkflux=true`, `ITensors.checkflux` is run on each resulting MPO tensor.
"""
function instantiate_MPO(
  offsets::Vector{Vector{Int}},
  block_sparse_matrices::Vector{Vector{BlockSparseMatrix{ValType}}},
  sites::Vector{<:Index},
  llinks::Vector{<:Index};
  splitblocks::Bool,
  checkflux::Bool,
)::MPO where {ValType<:Number}
  H = MPO(sites)

  @timeit "to_ITensor" Threads.@threads for n in 1:length(sites)
    H[n] = to_ITensor(
      offsets[n], block_sparse_matrices[n], llinks[n], llinks[n + 1], sites[n]; splitblocks
    )
  end

  # Remove the dummy link indices from the left and right.
  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(dag(llinks[end]))
  R[1] = 1.0

  H[1] *= L
  H[length(sites)] *= R

  if checkflux
    @timeit "checkflux" Threads.@threads for n in 1:length(sites)
      ITensors.checkflux(H[n])
    end
  end

  return H
end

@doc """
    MPO_new(ValType, os::OpIDSum, sites; basis_op_cache_vec=nothing,
      check_for_errors=true, checkflux=true, splitblocks=true, output_level=0,
      kwargs...) -> MPO

Construct an MPO from an `OpIDSum`.

Before construction, `os` can optionally be rewritten into the basis defined by
`basis_op_cache_vec`, and basic consistency checks can be run with
`check_for_errors=true`.

Keyword arguments:
- `basis_op_cache_vec=nothing`: A list of operators to use as a basis for each site.
  Products of operators on the same site are expressed as one of these basis operators. If `nothing`,
  a basis is inferred from the input and no basis transformation occurs.
- `check_for_errors=true`: Check the input `OpIDSum` for errors. This can be expensive
  for larger problems.
- `checkflux=true`: Check that the resulting MPO tensors all have a well-defined flux.
  This can be expensive for larger problems when `splitblocks=true`.
- `splitblocks=true`: Split the QN sectors into blocks of size one. This affects
  the sparsity of the resulting MPO, but can also slow down construction.
$_resume_MPO_kwargs
"""
function MPO_new(
  ValType::Type{<:Number},
  os::OpIDSum,
  sites::Vector{<:Index};
  basis_op_cache_vec=nothing,
  check_for_errors::Bool=true,
  checkflux::Bool=true,
  splitblocks::Union{Bool,Nothing}=nothing,
  output_level::Int=0,
  kwargs...,
)::MPO
  if isnothing(splitblocks) # TODO: Remove warning some time after v0.2.1 release
    splitblocks = true
    hasqns(sites) && Base.depwarn(
      "`splitblocks` not specified. The default is `true`, which is a change from prior behavior.",
      :MPO_new;
      force=true,
    )
  end

  prepare_opID_sum!(os, to_OpCacheVec(sites, basis_op_cache_vec))
  check_for_errors && check_os_for_errors(os)

  label = "Constructing MPOGraph from $(length(os)) terms"
  @time_if output_level 0 label g = MPOGraph(os)

  llinks = Vector{Index}(undef, length(sites) + 1)
  if hasqns(sites)
    llinks[1] = Index(QN() => 1; tags="Link,l=0", dir=ITensors.Out)
  else
    llinks[1] = Index(1; tags="Link,l=0")
  end

  offsets = Vector{Vector{Int}}(undef, length(sites))
  block_sparse_matrices = Vector{Vector{BlockSparseMatrix{ValType}}}(undef, length(sites))

  @time_if output_level 0 "Constructing MPO terms" resume_MPO_construction!(
    1,
    offsets,
    block_sparse_matrices,
    sites,
    llinks,
    g,
    os.op_cache_vec;
    output_level,
    kwargs...,
  )

  @time_if output_level 0 "Converting to ITensors" H = instantiate_MPO(
    offsets, block_sparse_matrices, sites, llinks; splitblocks, checkflux
  )

  return H
end

"""
    MPO_new(os::OpIDSum, sites; kwargs...) -> MPO

Construct an MPO from `os`, automatically choosing `Float64` or `ComplexF64`
from its coefficients and cached local operator matrices.
"""
function MPO_new(os::OpIDSum, sites::Vector{<:Index}; kwargs...)::MPO
  return MPO_new(determine_val_type(os), os, sites; kwargs...)
end

"""
    MPO_new(ValType, os::OpSum, sites; kwargs...) -> MPO

Convert an ITensor `OpSum` to `OpIDSum` form and construct an MPO with element
type `ValType`. Keyword arguments are forwarded to the `OpIDSum` method.
"""
function MPO_new(ValType::Type{<:Number}, os::OpSum, sites::Vector{<:Index}; kwargs...)::MPO
  return MPO_new(ValType, op_sum_to_opID_sum(os, sites), sites; kwargs...)
end

"""
    MPO_new(os::OpSum, sites; kwargs...) -> MPO

Convert an ITensor `OpSum` to `OpIDSum` form and construct an MPO, inferring the
numeric element type automatically.
"""
function MPO_new(os::OpSum, sites::Vector{<:Index}; kwargs...)::MPO
  return MPO_new(op_sum_to_opID_sum(os, sites), sites; kwargs...)
end

"""
    sparsity(mpo::MPO) -> Float64

Return the fraction of tensor entries in `mpo` that are structural zeros.

For dense tensors this is based on the full tensor size; for block-sparse
tensors `ITensors.nnz` counts stored entries, so entries omitted by block
sparsity are treated as structural zeros.
"""
function sparsity(mpo::MPO)::Float64
  num_entries = 0
  num_zeros = 0
  for tensor in mpo
    num_entries += prod(size(tensor))
    num_zeros += prod(size(tensor)) - ITensors.nnz(tensor)
  end

  return num_zeros / num_entries
end

"""
    block2_nnz(mpo::MPO) -> Tuple{Int,Int}

Count link-space blocks in `mpo` that are structural zeros.

When the MPO tensors are viewed as a matrix of operators, this
returns the total number of entries in the MPO and the total number
of structural nonzeros. This can be used to directly compare sparsities
with the `block2` storage format. The returned tuple is
`(total_link_blocks, nonzero_link_blocks)`.
"""
function block2_nnz(mpo::MPO)::Tuple{Int,Int}
  total_blocks = 0
  nnz_blocks = 0
  sites = noprime(siteinds(first, mpo))
  for (i, t) in enumerate(mpo)
    t = t.tensor
    total_blocks += prod(size(t)) ÷ dim(sites[i])^2

    link_locs = [j for j in 1:ndims(t) if inds(t)[j] ∉ (dag(sites[i]), sites[i]')]
    @assert length(link_locs) ∈ (1, 2)

    nz_link_coords = Set{Tuple{Int,Int}}()

    for b in ITensors.eachnzblock(t)
      block = ITensors.blockview(t, b)
      blockStart = NDTensors.blockstart(t, b) .- 1

      for coords in CartesianIndices(block)
        value = block[coords]
        value == 0 && continue

        coords = coords.I .+ blockStart

        link_coords = coords[link_locs]
        length(link_coords) == 1 && (link_coords = (only(link_coords), 0))
        push!(nz_link_coords, link_coords)
      end
    end

    nnz_blocks += length(nz_link_coords)
  end

  return total_blocks, nnz_blocks
end
