function my_ITensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{BlockSparseMatrix{C}},
  inds...;
  tol=0.0,
  checkflux=true,
)::ITensor where {C}
  is = Tuple(ITensors.indices(inds))
  blocks = Block{length(is)}[]
  T = ITensors.BlockSparseTensor(C, blocks, inds)
  _copyto_dropzeros!(T, offsets, block_sparse_matrices; tol)
  if checkflux
    ITensors.checkflux(T)
  end
  return itensor(T)
end

function _copyto_dropzeros!(T::ITensors.Tensor, offsets::Vector{Int}, block_sparse_matrices::Vector{<:BlockSparseMatrix}; tol)
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

  return T
end

@timeit function svdMPO_new(
  ValType::Type{<:Number},
  os::OpIDSum,
  op_cache_vec::OpCacheVec,
  sites::Vector{<:Index};
  tol::Real=-1,
  output_level::Int=0
)::MPO
  # TODO: This should be fixed.
  @assert !ITensors.using_auto_fermion()

  N = length(sites)

  @time_if output_level 0 "Constructing MPOGraph" g = MPOGraph(os, op_cache_vec)

  H = MPO(sites)

  llinks = Vector{Index}(undef, N + 1)
  if hasqns(sites)
    llinks[1] = Index(QN() => 1; tags="Link,l=0", dir=ITensors.Out)
  else
    llinks[1] = Index(1; tags="Link,l=0")
  end

  for n in 1:N
    output_level > 0 && println("At site $n/$(length(sites))")
    @time_if output_level 1 "at_site!" g, offsets, block_sparse_matrices, llinks[n + 1] = at_site!(
      ValType, g, n, sites, tol, op_cache_vec; output_level
    )

    # Constructing the tensor from an array is much faster than setting the components of the ITensor directly.
    # Especially for sparse tensors, and works for both sparse and dense tensors.
    @timeit "Constructing ITensor" let
      if hasqns(sites)
        H[n] = my_ITensor(
          offsets,
          block_sparse_matrices,
          dag(llinks[n]),
          llinks[n + 1],
          prime(sites[n]),
          dag(sites[n]);
          tol=1e-10,
          checkflux=false,
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
          tol=1e-10,
          checkflux=false,
        )
      end
    end
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

function MPO_new(
  ValType::Type{<:Number},
  os::OpIDSum,
  sites::Vector{<:Index},
  op_cache_vec::OpCacheVec;
  basis_op_cache_vec=nothing,
  tol::Real=-1,
  output_level::Int=0,
)::MPO
  op_cache_vec = to_OpCacheVec(sites, op_cache_vec)
  basis_op_cache_vec = to_OpCacheVec(sites, basis_op_cache_vec)
  os, op_cache_vec = prepare_opID_sum!(os, sites, op_cache_vec, basis_op_cache_vec)
  return svdMPO_new(ValType, os, op_cache_vec, sites; tol, output_level)
end

function MPO_new(
  os::OpIDSum,
  sites::Vector{<:Index},
  op_cache_vec;
  basis_op_cache_vec=nothing,
  tol::Real=-1,
  output_level::Int=0,
)::MPO
  op_cache_vec = to_OpCacheVec(sites, op_cache_vec)
  ValType = determine_val_type(os, op_cache_vec)
  return MPO_new(
    ValType, os, sites, op_cache_vec; basis_op_cache_vec, tol, output_level
  )
end

function MPO_new(
  ValType::Type{<:Number},
  os::OpSum,
  sites::Vector{<:Index};
  tol::Real=-1,
  output_level::Int=0,
)::MPO
  opID_sum, op_cache_vec = op_sum_to_opID_sum(os, sites)
  return MPO_new(
    ValType, opID_sum, sites, op_cache_vec; basis_op_cache_vec, tol, output_level
  )
end

function MPO_new(
  os::OpSum, sites::Vector{<:Index}; basis_op_cache_vec=nothing, tol::Real=-1, output_level::Int=0
)::MPO
  opID_sum, op_cache_vec = op_sum_to_opID_sum(os, sites)
  return MPO_new(
    opID_sum, sites, op_cache_vec; basis_op_cache_vec, tol, output_level
  )
end

function sparsity(mpo::MPO)::Float64
  num_entries = 0
  num_zeros = 0
  for tensor in mpo
    num_entries += prod(size(tensor))
    num_zeros += prod(size(tensor)) - ITensors.nnz(tensor)
  end

  return num_zeros / num_entries
end
