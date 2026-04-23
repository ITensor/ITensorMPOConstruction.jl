"""
    _qn_position_map(ind::ITensors.QNIndex) -> Tuple{Vector{Int},Vector{Int}}

Map dense index positions of a QN index to block coordinates.

For each 1-based dense position in `ind`, the first returned vector stores the
QN block number containing that position and the second returned vector stores
the offset within that block. These lookup tables are used when translating
dense link and site coordinates into `BlockSparseTensor` block coordinates.
"""
function _qn_position_map(ind::ITensors.QNIndex)::Tuple{Vector{Int},Vector{Int}}
  block_numbers = Vector{Int}(undef, dim(ind))
  block_offsets = Vector{Int}(undef, dim(ind))

  position = 1
  for block_number in 1:nblocks(ind)
    block_size = blockdim(ind, block_number)
    for block_offset in 1:block_size
      block_numbers[position] = block_number
      block_offsets[position] = block_offset
      position += 1
    end
  end

  return block_numbers, block_offsets
end

"""
    _blockid(left_block, right_block, site_prime_block, site_block,
             num_right_blocks, num_site_prime_blocks, num_site_blocks) -> Int

Encode a 4-index block coordinate into a dense 1-based block id.

The id orders blocks lexicographically with `left_block` slowest and
`site_block` fastest. It is only a temporary lookup key; `_block_from_id`
performs the inverse conversion.
"""
@inline function _blockid(
  left_block::Int,
  right_block::Int,
  site_prime_block::Int,
  site_block::Int,
  num_right_blocks::Int,
  num_site_prime_blocks::Int,
  num_site_blocks::Int,
)::Int
  return (
    ((left_block - 1) * num_right_blocks + (right_block - 1)) * num_site_prime_blocks +
    (site_prime_block - 1)
  ) * num_site_blocks + site_block
end

"""
    _block_from_id(block_id, num_right_blocks, num_site_prime_blocks, num_site_blocks)
        -> Block{4}

Decode the dense block id produced by `_blockid` into an ITensors `Block{4}`.
"""
@inline function _block_from_id(
  block_id::Int, num_right_blocks::Int, num_site_prime_blocks::Int, num_site_blocks::Int
)::Block{4}
  block_value = block_id - 1

  site_block = (block_value % num_site_blocks) + 1
  block_value ÷= num_site_blocks

  site_prime_block = (block_value % num_site_prime_blocks) + 1
  block_value ÷= num_site_prime_blocks

  right_block = (block_value % num_right_blocks) + 1
  left_block = (block_value ÷ num_right_blocks) + 1

  return Block((left_block, right_block, site_prime_block, site_block))
end

"""
    _fill_splitblocks!(blocks, block_values, offsets, block_sparse_matrices) -> Nothing

Fill split-block destination blocks and values from component block matrices.

This helper is used after all QN indices have been split into one-dimensional
blocks. For every nonzero matrix entry, it writes the corresponding
`Block((left_link, right_link + offset, site_prime, site))` destination and the
entry value into the preallocated `blocks` and `block_values` arrays. Zero
entries are skipped.
"""
function _fill_splitblocks!(
  blocks::Vector{Block{4}},
  block_values::Vector{C},
  offsets::Vector{Int},
  block_sparse_matrices::Vector{Dict{Tuple{Int,Int},Matrix{C}}},
) where {C}
  entry = 0

  for (offset, matrix) in zip(offsets, block_sparse_matrices)
    for ((left_link, right_link), block) in matrix
      shifted_right_link = right_link + offset

      @inbounds for i in axes(block, 1), j in axes(block, 2)
        value = block[i, j]
        iszero(value) && continue

        entry += 1
        blocks[entry] = Block((left_link, shifted_right_link, i, j))
        block_values[entry] = value
      end
    end
  end

  return nothing
end

"""
    _to_ITensor_splitblocks(offsets, block_sparse_matrices, llink, rlink, site) -> ITensor

Construct a QN-conserving MPO tensor using split one-dimensional QN blocks.

All three input indices are first passed through `ITensors.splitblocks`. Each
nonzero scalar entry in `block_sparse_matrices` then becomes one structural
`Block{4}` entry in the output tensor. `offsets[i]` places component `i` into
its slice of the outgoing right-link space, so local component right-link ids
are shifted to global right-link ids by `right_link + offsets[i]`.
"""
function _to_ITensor_splitblocks(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{Dict{Tuple{Int,Int},Matrix{C}}},
  llink::ITensors.QNIndex,
  rlink::ITensors.QNIndex,
  site::ITensors.QNIndex;
)::ITensor where {C}
  llink = ITensors.splitblocks(llink)
  rlink = ITensors.splitblocks(rlink)
  site = ITensors.splitblocks(site)

  inds = (dag(llink), rlink, prime(site), dag(site))

  num_nonzero_entries = sum(
    count(value -> !iszero(value), block) for
    matrix in block_sparse_matrices for block in values(matrix)
  )

  blocks = Vector{Block{4}}(undef, num_nonzero_entries)
  block_values = Vector{C}(undef, num_nonzero_entries)

  _fill_splitblocks!(blocks, block_values, offsets, block_sparse_matrices)

  block_offsets = ITensors.NDTensors.BlockOffsets{4}(
    blocks, 0:(num_nonzero_entries - 1)
  )

  T = ITensors.tensor(ITensors.NDTensors.BlockSparse(block_values, block_offsets), inds)
  return itensor(T)
end

"""
    to_ITensor(offsets, block_sparse_matrices, llink::QNIndex, rlink::QNIndex, site::QNIndex;
               splitblocks=false) -> ITensor

Assemble a block-sparse ITensor MPO tensor from component-local block matrices.

`block_sparse_matrices[i]` stores the local MPO matrix blocks for component `i`
as `(left_link, right_link) => local_operator_matrix`. The corresponding
`offsets[i]` shifts those component-local right-link ids into the global
outgoing bond space. The returned tensor has indices
`(dag(llink), rlink, prime(site), dag(site))`.

When `splitblocks=true`, construction delegates to `_to_ITensor_splitblocks`.
Otherwise the method discovers the QN blocks touched by nonzero entries,
allocates only those structural blocks, zero-initializes their storage, and
writes nonzero entries into the appropriate block views.
"""
function to_ITensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{Dict{Tuple{Int,Int},Matrix{C}}},
  llink::ITensors.QNIndex,
  rlink::ITensors.QNIndex,
  site::ITensors.QNIndex;
  splitblocks::Bool=false,
)::ITensor where {C}
  if splitblocks
    return _to_ITensor_splitblocks(
      offsets, block_sparse_matrices, llink, rlink, site
    )
  end

  inds = (dag(llink), rlink, prime(site), dag(site))

  left_blocks, left_offsets = _qn_position_map(first(inds))
  right_blocks, right_offsets = _qn_position_map(inds[2])
  site_prime_blocks, site_prime_offsets = _qn_position_map(inds[3])
  site_blocks, site_offsets = _qn_position_map(inds[4])

  num_right_blocks = nblocks(inds[2])
  num_site_prime_blocks = nblocks(inds[3])
  num_site_blocks = nblocks(inds[4])

  block_ids = Set{Int}()
  sizehint!(block_ids, sum(length, block_sparse_matrices))

  for (offset, matrix) in zip(offsets, block_sparse_matrices)
    for ((left_link, right_link), block) in matrix
      shifted_right_link = right_link + offset
      left_block = left_blocks[left_link]
      right_block = right_blocks[shifted_right_link]

      @inbounds for i in axes(block, 1), j in axes(block, 2)
        value = block[i, j]
        iszero(value) && continue

        push!(
          block_ids,
          _blockid(
            left_block,
            right_block,
            site_prime_blocks[i],
            site_blocks[j],
            num_right_blocks,
            num_site_prime_blocks,
            num_site_blocks,
          ),
        )
      end
    end
  end

  isempty(block_ids) && return itensor(ITensors.BlockSparseTensor(C, Block{4}[], inds))

  block_id_list = collect(block_ids)
  blocks = Vector{Block{4}}(undef, length(block_id_list))
  for n in eachindex(block_id_list)
    blocks[n] = _block_from_id(
      block_id_list[n], num_right_blocks, num_site_prime_blocks, num_site_blocks
    )
  end

  T = ITensors.BlockSparseTensor(C, undef, blocks, inds)

  first_view = array(ITensors.blockview(T, first(blocks)))
  fill!(first_view, zero(C))
  ViewT = typeof(first_view)
  block_views = sizehint!(Dict{Int,ViewT}(), length(block_id_list))
  block_views[first(block_id_list)] = first_view
  for n in 2:length(block_id_list)
    block_views[block_id_list[n]] = array(ITensors.blockview(T, blocks[n]))
    fill!(block_views[block_id_list[n]], zero(C))
  end

  Threads.@threads for i in eachindex(block_sparse_matrices)
    offset = offsets[i]
    matrix = block_sparse_matrices[i]

    for ((left_link, right_link), block) in matrix
      shifted_right_link = right_link + offset
      left_block = left_blocks[left_link]
      right_block = right_blocks[shifted_right_link]
      left_offset = left_offsets[left_link]
      right_offset = right_offsets[shifted_right_link]

      @inbounds for i in axes(block, 1), j in axes(block, 2)
        value = block[i, j]
        iszero(value) && continue

        block_id = _blockid(
          left_block,
          right_block,
          site_prime_blocks[i],
          site_blocks[j],
          num_right_blocks,
          num_site_prime_blocks,
          num_site_blocks,
        )
        block_views[block_id][
          left_offset, right_offset, site_prime_offsets[i], site_offsets[j]
        ] = value
      end
    end
  end

  return itensor(T)
end

"""
    to_ITensor(offsets, block_sparse_matrices, llink::Index, rlink::Index, site::Index;
               splitblocks=false) -> ITensor

Assemble a dense-index ITensor MPO tensor from component-local block matrices.

This fallback is used for non-QN indices. It allocates a dense array with axes
`(llink, rlink, prime(site), site)` and writes each local operator matrix into
the slice selected by `(left_link, right_link + offsets[i], :, :)`.
The `splitblocks` keyword is accepted for interface compatibility but has no
effect for dense indices.
"""
function to_ITensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{BlockSparseMatrix{C}},
  llink::Index,
  rlink::Index,
  site::Index;
  splitblocks::Bool=false,
)::ITensor where {C}
  tensor = zeros(C, dim(llink), dim(rlink), dim(prime(site)), dim(site))
  Threads.@threads for i in eachindex(offsets)
    offset = offsets[i]
    matrix = block_sparse_matrices[i]
    for ((left_link, right_link), block) in matrix
      tensor[left_link, right_link + offset, :, :] = block
    end
  end

  return itensor(tensor, llink, rlink, prime(site), site)
end
