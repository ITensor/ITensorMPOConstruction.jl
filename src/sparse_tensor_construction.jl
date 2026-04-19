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

function to_ITensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{Dict{Tuple{Int,Int},Matrix{C}}},
  llink::ITensors.QNIndex,
  rlink::ITensors.QNIndex,
  site::ITensors.QNIndex;
  checkflux::Bool=true
)::ITensor where {C}
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

  isempty(block_ids) && return ITensors.BlockSparseTensor(C, Block{4}[], inds)

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

  for (offset, matrix) in zip(offsets, block_sparse_matrices)
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

  checkflux && ITensors.checkflux(T)

  return itensor(T)
end

function to_Itensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{BlockSparseMatrix{C}},
  llink::Index,
  rlink::Index,
  site::Index;
  checkflux::Bool=true
)::ITensor where {C}
  tensor = zeros(C, dim(llink), dim(rlink), dim(prime(site)), dim(site))
  for (offset, matrix) in zip(offsets, block_sparse_matrices)
    for ((left_link, right_link), block) in matrix
      tensor[left_link, right_link + offset, :, :] = block
    end
  end

  return itensor(tensor, llink, rlink, prime(site), site)
end
