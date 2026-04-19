function my_ITensor_old(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{Dict{Tuple{Int,Int},Matrix{C}}},
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