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
  my_copyto_dropzeros!(T, offsets, block_sparse_matrices; tol)
  if checkflux
    ITensors.checkflux(T)
  end
  return itensor(T)
end

function my_copyto_dropzeros!(T::ITensors.Tensor, offsets::Vector{Int}, block_sparse_matrices::Vector{<:BlockSparseMatrix}; tol)
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

function resume_svd_MPO(
  ValType::Type{<:Number},
  n_init::Int,
  H::MPO,
  sites::Vector{<:Index},
  llinks::Vector{<:Index},
  g::MPOGraph,
  op_cache_vec::OpCacheVec;
  tol::Real=1,
  absoluteTol::Bool=false,
  combine_qn_sectors::Bool=false,
  call_back::Function=(args...) -> nothing,
  output_level::Int=0
  )::MPO
  @assert !ITensors.using_auto_fermion() # TODO: This should be fixed.
  @assert tol >= 0

  N = length(sites)

  for n in n_init:N
    output_level > 0 && println("At site $n/$(length(sites)) the graph takes up $(Base.format_bytes(Base.summarysize(g)))")
    @time_if output_level 1 "at_site!" g, offsets, block_sparse_matrices, llinks[n + 1] = at_site!(
      ValType, g, n, sites, tol, absoluteTol, op_cache_vec; combine_qn_sectors, output_level
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

@timeit function svdMPO_new(
  ValType::Type{<:Number},
  os::OpIDSum,
  sites::Vector{<:Index};
  output_level::Int=0,
  kwargs...
)::MPO
  # TODO: This should be fixed.
  @assert !ITensors.using_auto_fermion()

  N = length(sites)

  @time_if output_level 0 "Constructing MPOGraph" g = MPOGraph(os)

  H = MPO(sites)

  llinks = Vector{Index}(undef, N + 1)
  if hasqns(sites)
    llinks[1] = Index(QN() => 1; tags="Link,l=0", dir=ITensors.Out)
  else
    llinks[1] = Index(1; tags="Link,l=0")
  end

  return resume_svd_MPO(ValType, 1, H, sites, llinks, g, os.op_cache_vec; output_level, kwargs...)
end

function MPO_new(
  ValType::Type{<:Number},
  os::OpIDSum,
  sites::Vector{<:Index};
  basis_op_cache_vec::Union{OpCacheVec, Nothing}=nothing,
  kwargs...,
)::MPO
  prepare_opID_sum!(os, basis_op_cache_vec)
  return svdMPO_new(ValType, os, sites; kwargs...)
end

function MPO_new(
  os::OpIDSum,
  sites::Vector{<:Index};
  kwargs...
)::MPO
  ValType = determine_val_type(os)
  return MPO_new(ValType, os, sites; kwargs...)
end

function MPO_new(
  ValType::Type{<:Number},
  os::OpSum,
  sites::Vector{<:Index};
  kwargs...
)::MPO
  opID_sum = op_sum_to_opID_sum(os, sites)
  return MPO_new(ValType, opID_sum, sites; kwargs...)
end

function MPO_new(
  os::OpSum,
  sites::Vector{<:Index};
  kwargs...
)::MPO
  opID_sum = op_sum_to_opID_sum(os, sites)
  return MPO_new(opID_sum, sites; kwargs...)
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

function redistribute_norm!(H::MPO)::Nothing
  norms = [norm(t) for t in H]

  min, max = argmin(norms), argmax(norms)
  while norms[max] > 2 * norms[min]
    H[min] *= 2
    norms[min] *= 2

    H[max] /= 2
    norms[max] /= 2

    min, max = argmin(norms), argmax(norms)
  end

  return nothing
end
