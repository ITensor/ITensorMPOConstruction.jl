# @timeit function at_site!(
#   ValType::Type{<:Number},
#   graphs::Dict{QN,MPOGraph{C}},
#   n::Int,
#   sites::Vector{<:Index},
#   tol::Real,
#   op_cache_vec::OpCacheVec,
# )::Tuple{Dict{QN,MPOGraph{C}},BlockSparseMatrix{ValType},Index} where {C}
#   has_qns = hasqns(sites)
#   next_graphs = Dict{QN,MPOGraph{C}}()

#   matrix = BlockSparseMatrix{ValType}()

#   qi = Vector{Pair{QN,Int}}()
#   outgoing_link_offset = 0

#   id_of_left_vertex_in_zero_flux_term = Vector{Int}()

#   for (in_qn, g) in graphs
#     W = sparse_edge_weights(g)
#     Q, R, prow, pcol, rank = sparse_qr(W, tol)

#     W = nothing

#     #=
#     If we are at the last site, then R will be a 1x1 matrix containing an overall scaling factor
#     that we need to account for.
#     =#
#     if n == length(sites)
#       Q *= only(R)
#     end

#     if has_qns
#       append!(qi, [in_qn => rank])
#     end

#     # Form the local transformation tensor.
#     # At the 0th dummy site we don't need to do this.
#     @timeit "constructing the sparse matrix" if n != 0
#       ## TODO: This is wastefull
#       local_matrices = [
#         my_matrix([lv.opOnSite], op_cache_vec[n]; needs_JW_string=lv.needs_JW_string) for
#         lv in g.left_vertex_values
#       ]

#       for_non_zeros(Q, left_size(g), rank) do weight, i, m
#         right_link = m + outgoing_link_offset
#         lv = left_value(g, prow[i])

#         matrix_element = get!(matrix, (lv.link, right_link)) do
#           return zeros(C, dim(sites[n]), dim(sites[n]))
#         end

#         matrix_element .+= weight * local_matrices[prow[i]]
#       end
#     end

#     ## Free up some memory
#     Q = prow = nothing

#     # Connect this output \tilde{A}_m to \tilde{B}_m = (R \vec{B})_m
#     # If we are processing the last site then there's no need to do this.
#     @timeit "constructing the next graphs" if n != length(sites)
#       next_graph_zero_flux = get!(next_graphs, in_qn) do
#         return MPOGraph{C}()
#       end

#       resize!(id_of_left_vertex_in_zero_flux_term, rank)
#       id_of_left_vertex_in_zero_flux_term .= 0
#       for_non_zeros_batch(R, right_size(g)) do weights, ms, j
#         j = pcol[j]

#         rv = right_value(g, j)
#         if !isempty(rv.ops) && rv.ops[end].n == n + 1
#           onsite = pop!(rv.ops)
#           flux = op_cache_vec[n + 1][onsite.id].qnFlux
#           is_fermionic = xor(rv.is_fermionic, op_cache_vec[n + 1][onsite.id].is_fermionic)

#           rv = RightVertex(rv.ops, is_fermionic)

#           next_graph = get!(next_graphs, in_qn + flux) do
#             return MPOGraph{C}()
#           end

#           add_edges!(next_graph, rv, rank, ms, outgoing_link_offset, onsite, weights)
#         else
#           add_edges_vector_lookup!(
#             next_graph_zero_flux,
#             rv,
#             rank,
#             ms,
#             outgoing_link_offset,
#             OpID(1, n + 1),
#             weights,
#             id_of_left_vertex_in_zero_flux_term,
#           )
#         end
#       end

#       if num_edges(next_graph_zero_flux) == 0
#         delete!(next_graphs, in_qn)
#       end

#       for (_, next_graph) in next_graphs
#         empty!(next_graph.left_vertex_ids)
#       end
#     end

#     empty!(g)
#     outgoing_link_offset += rank
#   end

#   for (_, next_graph) in next_graphs
#     empty!(next_graph.right_vertex_ids)
#   end

#   if has_qns
#     outgoing_link = Index(qi...; tags="Link,l=$n", dir=ITensors.Out)
#   else
#     outgoing_link = Index(outgoing_link_offset; tags="Link,l=$n")
#   end

#   return next_graphs, matrix, outgoing_link
# end

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

  g = MPOGraph{6, C}(os, op_cache_vec[1])

  H = MPO(sites)

  llinks = Vector{Index}(undef, N + 1)
  if hasqns(sites)
    llinks[1] = Index(QN(); tags="Link,l=0", dir=ITensors.Out)
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
