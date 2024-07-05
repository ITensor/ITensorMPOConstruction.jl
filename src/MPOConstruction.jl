function svdMPO_new(
  ValType::Type{<:Number},
  os::OpIDSum{C},
  op_cache_vec::OpCacheVec,
  sites::Vector{<:Index};
  tol::Real=-1,
  verbose::Bool=false
)::MPO where {C}
  # TODO: This should be fixed.
  @assert !ITensors.using_auto_fermion()

  N = length(sites)

  g = MPOGraph{6, C}(os, op_cache_vec)

  H = MPO(sites)

  llinks = Vector{Index}(undef, N + 1)
  if hasqns(sites)
    llinks[1] = Index(QN() => 1; tags="Link,l=0", dir=ITensors.Out)
  else
    llinks[1] = Index(1; tags="Link,l=0")
  end

  for n in 1:N
    verbose && println("n = $n: at_site")
    g, block_sparse_matrix, llinks[n + 1] = at_site!(
      ValType, g, n, sites, tol, op_cache_vec
    )

    # Constructing the tensor from an array is much faster than setting the components of the ITensor directly.
    # Especially for sparse tensors, and works for both sparse and dense tensors.
    let
      tensor = zeros(
        ValType,
        dim(dag(llinks[n])),
        dim(llinks[n + 1]),
        dim(prime(sites[n])),
        dim(dag(sites[n])),
      )

      for ((left_link, right_link), block) in block_sparse_matrix
        tensor[left_link, right_link, :, :] = block
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
  verbose::Bool=false,
)::MPO
  op_cache_vec = to_OpCacheVec(sites, op_cache_vec)
  basis_op_cache_vec = to_OpCacheVec(sites, basis_op_cache_vec)
  os, op_cache_vec = prepare_opID_sum!(os, sites, op_cache_vec, basis_op_cache_vec)
  return svdMPO_new(ValType, os, op_cache_vec, sites; tol, verbose)
end

function MPO_new(
  os::OpIDSum,
  sites::Vector{<:Index},
  op_cache_vec;
  basis_op_cache_vec=nothing,
  tol::Real=-1,
  verbose::Bool=false,
)::MPO
  op_cache_vec = to_OpCacheVec(sites, op_cache_vec)
  ValType = determine_val_type(os, op_cache_vec)
  return MPO_new(
    ValType, os, sites, op_cache_vec; basis_op_cache_vec, tol, verbose
  )
end

function MPO_new(
  ValType::Type{<:Number},
  os::OpSum,
  sites::Vector{<:Index};
  tol::Real=-1,
  verbose::Bool=false,
)::MPO
  opID_sum, op_cache_vec = op_sum_to_opID_sum(os, sites)
  return MPO_new(
    ValType, opID_sum, sites, op_cache_vec; basis_op_cache_vec, tol, verbose
  )
end

function MPO_new(
  os::OpSum, sites::Vector{<:Index}; basis_op_cache_vec=nothing, tol::Real=-1, verbose::Bool=false
)::MPO
  opID_sum, op_cache_vec = op_sum_to_opID_sum(os, sites)
  return MPO_new(
    opID_sum, sites, op_cache_vec; basis_op_cache_vec, tol, verbose
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
