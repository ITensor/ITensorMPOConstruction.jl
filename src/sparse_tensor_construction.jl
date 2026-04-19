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
    to_sparse_itensor(offsets, block_sparse_matrices, inds...; tol=0.0, checkflux=true) -> ITensor

Assemble an `ITensor` from component-local block-sparse MPO data.

`offsets` gives the starting outgoing-link offset for each connected component,
and `block_sparse_matrices` stores the dense local blocks keyed by link-index
pairs. Entries with magnitude at most `tol` are dropped during the copy.
"""
function to_sparse_itensor(
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

  block_writes = [writes_by_block[block] for block in blocks]
  block_offsets = [NDTensors.offset(T, block) for block in blocks]
  block_dims = [NDTensors.blockdims(T, block) for block in blocks]
  storage_data = NDTensors.data(T)

  Threads.@threads for block_id in eachindex(blocks)
    block = blocks[block_id]
    row_range = site_in_block_ranges[Int(block[3])]
    col_range = site_out_block_ranges[Int(block[4])]
    block_offset = block_offsets[block_id]
    block_dim = prod(block_dims[block_id])
    block_data = reshape(
      @view(storage_data[(block_offset + 1):(block_offset + block_dim)]),
      block_dims[block_id],
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

  if checkflux
    ITensors.checkflux(T)
  end

  return itensor(T)
end

function to_dense_itensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{BlockSparseMatrix{C}},
  llink::Index,
  rlink::Index,
  site::Index
)::ITensor where {C}
  tensor = zeros(C, dim(llink), dim(rlink), dim(prime(site)), dim(site))
  for (offset, matrix) in zip(offsets, block_sparse_matrices)
    for ((left_link, right_link), block) in matrix
      tensor[left_link, right_link + offset, :, :] = block
    end
  end

  return itensor(tensor, llink, rlink, prime(site), site)
end