function _index_block_ranges(ind)
  ranges = Vector{UnitRange{Int}}(undef, nblocks(ind))
  block_start = 1
  for b in eachindex(ranges)
    block_stop = block_start + (hasqns(ind) ? blockdim(ind, b) : dim(ind)) - 1
    ranges[b] = block_start:block_stop
    block_start = block_stop + 1
  end
  return ranges
end

function _index_block_positions(ind)
  pos_to_block = Vector{Int}(undef, dim(ind))
  pos_to_offset = Vector{Int}(undef, dim(ind))

  for (block, range) in enumerate(_index_block_ranges(ind))
    for (offset, pos) in enumerate(range)
      pos_to_block[pos] = block
      pos_to_offset[pos] = offset
    end
  end

  return pos_to_block, pos_to_offset
end

function _any_above_tol(
  matrix::AbstractMatrix, row_range::UnitRange{Int}, col_range::UnitRange{Int}, tol::Real
)::Bool
  @inbounds for j in col_range
    for i in row_range
      abs(matrix[i, j]) > tol && return true
    end
  end
  return false
end

function _copy_above_tol!(dest, src, tol::Real)
  @inbounds for j in axes(src, 2)
    for i in axes(src, 1)
      value = src[i, j]
      abs(value) > tol || continue
      dest[i, j] = value
    end
  end
  return dest
end

"""
    my_ITensor(offsets, block_sparse_matrices, inds...; tol=0.0, checkflux=true) -> ITensor

Assemble an `ITensor` from component-local block-sparse MPO data.

`offsets` gives the starting outgoing-link offset for each connected component,
and `block_sparse_matrices` stores the dense local blocks keyed by link-index
pairs. Entries with magnitude at most `tol` are dropped during the copy.
"""
function my_ITensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{BlockSparseMatrix{C}},
  inds...;
  tol=0.0,
  checkflux=true,
)::ITensor where {C}
  @assert length(inds) == 4

  left_block_of_pos, left_offset_of_pos = _index_block_positions(inds[1])
  right_block_of_pos, right_offset_of_pos = _index_block_positions(inds[2])
  site_in_block_ranges = _index_block_ranges(inds[3])
  site_out_block_ranges = _index_block_ranges(inds[4])

  site_block_specs = Tuple{Int,Int,UnitRange{Int},UnitRange{Int}}[]
  for site_in_block in eachindex(site_in_block_ranges)
    for site_out_block in eachindex(site_out_block_ranges)
      push!(
        site_block_specs,
        (
          site_in_block,
          site_out_block,
          site_in_block_ranges[site_in_block],
          site_out_block_ranges[site_out_block],
        ),
      )
    end
  end

  writes_by_block = Dict{Block{4},Vector{Tuple{Int,Int,Matrix{C}}}}()
  for (offset, matrix) in zip(offsets, block_sparse_matrices)
    for ((left_link, right_link), local_matrix) in matrix
      left_block = left_block_of_pos[left_link]
      left_offset = left_offset_of_pos[left_link]
      right_pos = right_link + offset
      right_block = right_block_of_pos[right_pos]
      right_offset = right_offset_of_pos[right_pos]

      for (site_in_block, site_out_block, row_range, col_range) in site_block_specs
        _any_above_tol(local_matrix, row_range, col_range, tol) || continue
        block = Block(left_block, right_block, site_in_block, site_out_block)
        push!(
          get!(writes_by_block, block) do
            Tuple{Int,Int,Matrix{C}}[]
          end,
          (left_offset, right_offset, local_matrix),
        )
      end
    end
  end

  blocks = sort!(collect(keys(writes_by_block)))
  T = ITensors.BlockSparseTensor(C, undef, blocks, inds)

  if !isempty(blocks)
    block_writes = [writes_by_block[block] for block in blocks]
    block_offsets = [NDTensors.offset(T, block) for block in blocks]
    block_dims = [NDTensors.blockdims(T, block) for block in blocks]
    storage_data = NDTensors.data(T)

    # Each iteration owns one preallocated tensor block, so threaded writes stay
    # on disjoint slices of the backing storage and never mutate block metadata.
    use_threads = Threads.nthreads() > 1 && length(blocks) >= 4 * Threads.nthreads()

    if use_threads
      Threads.@threads for block_id in eachindex(blocks)
        block = blocks[block_id]
        row_range = site_in_block_ranges[Int(block[3])]
        col_range = site_out_block_ranges[Int(block[4])]
        block_offset = block_offsets[block_id]
        block_dim = prod(block_dims[block_id])
        block_data = reshape(
          @view(storage_data[(block_offset + 1):(block_offset + block_dim)]), block_dims[block_id]
        )
        fill!(block_data, zero(C))

        for (left_offset, right_offset, local_matrix) in block_writes[block_id]
          local_block = @view local_matrix[row_range, col_range]
          if iszero(tol)
            copyto!(@view(block_data[left_offset, right_offset, :, :]), local_block)
          else
            _copy_above_tol!(
              @view(block_data[left_offset, right_offset, :, :]), local_block, tol
            )
          end
        end
      end
    else
      for block_id in eachindex(blocks)
        block = blocks[block_id]
        row_range = site_in_block_ranges[Int(block[3])]
        col_range = site_out_block_ranges[Int(block[4])]
        block_offset = block_offsets[block_id]
        block_dim = prod(block_dims[block_id])
        block_data = reshape(
          @view(storage_data[(block_offset + 1):(block_offset + block_dim)]), block_dims[block_id]
        )
        fill!(block_data, zero(C))

        for (left_offset, right_offset, local_matrix) in block_writes[block_id]
          local_block = @view local_matrix[row_range, col_range]
          if iszero(tol)
            copyto!(@view(block_data[left_offset, right_offset, :, :]), local_block)
          else
            _copy_above_tol!(
              @view(block_data[left_offset, right_offset, :, :]), local_block, tol
            )
          end
        end
      end
    end
  end

  if checkflux
    ITensors.checkflux(T)
  end

  return itensor(T)
end

function my_ITensor_old(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{BlockSparseMatrix{C}},
  inds...;
  tol=0.0,
  checkflux=true,
)::ITensor where {C}
  T = ITensors.BlockSparseTensor(C, Block{length(inds)}[], inds)

  for (offset, matrix) in zip(offsets, block_sparse_matrices)
    for ((left_link, right_link), block) in matrix
      for i in 1:size(block, 1)
        for j in 1:size(block, 2)
          if abs(block[i, j]) > tol
            T[left_link, right_link + offset, i, j] = block[i, j]
          end
        end
      end
    end
  end

  if checkflux
    ITensors.checkflux(T)
  end

  return itensor(T)
end

_resume_MPO_kwargs = """
- `tol=1`: A multiplicative modifier to the default tolerance used in the SPQR,
  see [SPQR user guide Section 2.3](https://fossies.org/linux/SuiteSparse/SPQR/Doc/spqr_user_guide.pdf).
  The value of the default tolerance depends on the input matrix, which means a different
  tolerance is used for each decomposition. In the cases we have examined, the default 
  tolerance works great for producing accurate MPOs.
- `absolute_tol=false`: If true, override the default adaptive tolerance scheme outlined above,
  and use the value of `tol` as the single tolerance for each decomposition.
- `combine_qn_sectors=false`: When `true`, the blocks of the MPO corresponding to the same
  quantum numbers are merged together into a single block. This can decrease the resulting sparsity.
- `call_back`: A function that is called after constructing the MPO tensor at `cur_site`.
  Called as `call_back(cur_site, H, sites, llinks, cur_graph, op_cache_vec)`.
  Primarily used for writing checkpoints to disk for large calculations.
- `output_level=0`: Controls progress and timing output.
"""

@doc """
    resume_MPO_construction(ValType, n_init, H, sites, llinks, g, op_cache_vec;
      tol=1, absolute_tol=false, combine_qn_sectors=false, call_back=..., output_level=0) -> MPO

Continue MPO construction starting from site `n_init` using the current graph
state `g`.

This is the low-level driver underlying `MPO_new`. At each site it factors the
current graph with `at_site!`, builds the local MPO tensor, stores it into `H`, 
and advances to the next-site graph. The callback is invoked after each site is completed.

Keyword arguments:
$_resume_MPO_kwargs
"""
function resume_MPO_construction(
  ValType::Type{<:Number},
  n_init::Int,
  H::MPO,
  sites::Vector{<:Index},
  llinks::Vector{<:Index},
  g::MPOGraph,
  op_cache_vec::OpCacheVec;
  tol::Real=1,
  absolute_tol::Bool=false,
  combine_qn_sectors::Bool=false,
  call_back::Function=(
    cur_site::Int,
    H::MPO,
    sites::Vector{<:Index},
    llinks::Vector{<:Index},
    cur_graph::MPOGraph,
    op_cache_vec::OpCacheVec,
  ) -> nothing,
  output_level::Int=0,
)::MPO
  @assert !ITensors.using_auto_fermion() # TODO: This should be fixed.
  @assert tol >= 0

  N = length(sites)

  for n in n_init:N
    output_level > 0 && println(
      "At site $n/$(length(sites)) the graph takes up $(Base.format_bytes(Base.summarysize(g)))",
    )
    @time_if output_level 1 "at_site!" g, offsets, block_sparse_matrices, llinks[n + 1] = at_site!(
      ValType,
      g,
      n,
      sites,
      tol,
      absolute_tol,
      op_cache_vec;
      combine_qn_sectors,
      output_level,
    )

    @timeit "Constructing ITensor" let
      if hasqns(sites)
        H[n] = my_ITensor(
          offsets,
          block_sparse_matrices,
          dag(llinks[n]),
          llinks[n + 1],
          prime(sites[n]),
          dag(sites[n]);
          tol=0.0,
          checkflux=true,
        )
      else
        tensor = zeros(
          ValType,
          dim(dag(llinks[n])),
          dim(llinks[n + 1]),
          dim(prime(sites[n])),
          dim(dag(sites[n])),
        )

        for (offset, matrix) in zip(offsets, block_sparse_matrices)
          for ((left_link, right_link), block) in matrix
            tensor[left_link, right_link + offset, :, :] = block
          end
        end

        H[n] = itensor(
          tensor,
          dag(llinks[n]),
          llinks[n + 1],
          prime(sites[n]),
          dag(sites[n]);
          tol=0.0,
          checkflux=true,
        )
      end
    end

    call_back(n, H, sites, llinks, g, op_cache_vec)
  end

  # Remove the dummy link indices from the left and right.
  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(dag(llinks[N + 1]))
  R[1] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end

@doc """
    MPO_new(ValType, os::OpIDSum, sites; basis_op_cache_vec=nothing,
      check_for_errors=true, output_level=0, kwargs...) -> MPO

Construct an MPO from a `OpIDSum`.

Before construction, `os` can optionally be rewritten into the basis defined by
`basis_op_cache_vec`, and basic consistency checks can be run with `check_for_errors=true`.

Keyword arguments:
- `basis_op_cache_vec=nothing`: A list of operators to use as a basis for each site.
  The operators on each site are expressed as one of these basis operators. If `nothing`
  a basis is inferred from the input and no basis transformation occurs.
- `check_for_errors=true`: Check the input OpSum for errors, this can be expensive
  for larger problems.
$_resume_MPO_kwargs
"""
function MPO_new(
  ValType::Type{<:Number},
  os::OpIDSum,
  sites::Vector{<:Index};
  basis_op_cache_vec=nothing,
  check_for_errors::Bool=true,
  output_level::Int=0,
  kwargs...,
)::MPO
  prepare_opID_sum!(os, to_OpCacheVec(sites, basis_op_cache_vec))
  check_for_errors && check_os_for_errors(os)

  @time_if output_level 0 "Constructing MPOGraph" g = MPOGraph(os)

  H = MPO(sites)

  llinks = Vector{Index}(undef, length(sites) + 1)
  if hasqns(sites)
    llinks[1] = Index(QN() => 1; tags="Link,l=0", dir=ITensors.Out)
  else
    llinks[1] = Index(1; tags="Link,l=0")
  end

  return resume_MPO_construction(
    ValType, 1, H, sites, llinks, g, os.op_cache_vec; output_level, kwargs...
  )
end

"""
    MPO_new(os::OpIDSum, sites; kwargs...) -> MPO

Construct an MPO from `os`, automatically choosing `Float64` or `ComplexF64`.
"""
function MPO_new(os::OpIDSum, sites::Vector{<:Index}; kwargs...)::MPO
  return MPO_new(determine_val_type(os), os, sites; kwargs...)
end

"""
    MPO_new(ValType, os::OpSum, sites; kwargs...) -> MPO

Convert an ITensor `OpSum` to `OpIDSum` form and construct an MPO with element
type `ValType`.
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

When the MPO tensors are viewed as a matrix-op-operators, this
returns the total number of entries in the MPO and the total number
of structural non-zeros. This can be used to directly compare sparsities
with the `block2` storage format.
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
