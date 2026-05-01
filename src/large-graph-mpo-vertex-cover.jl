
function _signed_symbolic_local_op_id(lv::LeftVertex, ::Type{Ti})::Ti where {Ti<:Integer}
  signed_local_op_id = lv.needs_JW_string ? -Int(lv.op_id) : Int(lv.op_id)
  return Ti(signed_local_op_id)
end

function _add_vertex_cover_term!(
  matrix::BlockSparseMatrix{Matrix{ValType}},
  m::Int,
  lv::LeftVertex,
  weight::Number,
  op_cache::Vector{OpInfo},
  site_dim::Int,
)::Nothing where {ValType}
  local_op = op_cache[lv.op_id].matrix
  matrix_element = get!(matrix[m], lv.link) do
    zeros(ValType, site_dim, site_dim)
  end
  add_to_local_matrix!(matrix_element, weight, local_op, lv.needs_JW_string)
  return nothing
end

function _add_vertex_cover_term!(
  matrix::BlockSparseMatrix{SymbolicLocalMatrix{W, Ti}},
  m::Int,
  lv::LeftVertex,
  weight::W,
  op_cache::Vector{OpInfo},
  site_dim::Int,
)::Nothing where {W, Ti}
  terms = get!(matrix[m], lv.link) do
    SymbolicLocalMatrix{W, Ti}()
  end
  _append_symbolic_local_matrix_term!(
    terms, weight, _signed_symbolic_local_op_id(lv, Ti)
  )
  return nothing
end

"""
    process_vertex_cover!(
      matrix_of_cc, rank_of_cc, next_edges_of_cc, g, ccs, n, sites, op_cache_vec
    ) -> Nothing

Process every connected component using the minimum-vertex-cover specialization.

For each component, the cover columns are laid out as `[left_cover; right_cover]`
in the local bond dimension. Left-cover columns emit single local operators with
unit graph weight; right-cover columns accumulate uncovered edges with their
stored edge weights. On the final site, only the local tensor blocks are needed;
otherwise this also builds `next_edges_of_cc` for the graph at site `n + 1`.
"""
@timeit function process_vertex_cover!(
  matrix_of_cc::Vector{BlockSparseMatrix{MatrixType}},
  rank_of_cc::Vector{Int},
  next_edges_of_cc::Vector{Matrix{Tuple{Vector{Int},Vector{C}}}},
  g::MPOGraph{N,C,Ti},
  ccs::BipartiteGraphConnectedComponents,
  n::Int,
  sites::Vector{<:Index},
  op_cache_vec::OpCacheVec,
)::Nothing where {MatrixType,N,C,Ti}
  site_dim = dim(sites[n])
  op_cache = op_cache_vec[n]

  next_op_of_rv_id = Ti[]
  if n != length(sites)
    resize!(next_op_of_rv_id, right_size(g))
    Threads.@threads for rv_id in 1:right_size(g)
      next_op_of_rv_id[rv_id] = get_onsite_op(right_vertex(g, rv_id), n + 1)
    end
  end

  Threads.@threads for cc in 1:num_connected_components(ccs)
    lvs_of_component::Vector{Int} = ccs.lvs_of_component[cc]
    position_of_rvs_in_component = ccs.position_of_rvs_in_component
    rv_size_of_component = ccs.rv_size_of_component[cc]

    ## No idea why, but these need to be typed or allocations go nuts.
    left_cover::Vector{Int}, right_cover::Vector{Int} = minimum_vertex_cover(g, ccs, cc)

    rank = length(left_cover) + length(right_cover)
    rank_of_cc[cc] = rank

    matrix = BlockSparseMatrix(MatrixType, rank)
    matrix_of_cc[cc] = matrix

    ## Construct the tensor from the left cover.
    @inbounds for m in eachindex(left_cover)
      lv_id = lvs_of_component[left_cover[m]]
      lv = left_vertex(g, lv_id)

      _add_vertex_cover_term!(matrix, m, lv, one(C), op_cache, site_dim)
    end

    ## Construct the tensor from the right cover.
    let
      in_left_cover = falses(length(lvs_of_component))
      @inbounds for local_id in left_cover
        in_left_cover[local_id] = true
      end

      uncovered_left_ids = Vector{Int}(undef, length(lvs_of_component) - length(left_cover))
      next_uncovered = 1
      @inbounds for local_id in eachindex(lvs_of_component)
        if !in_left_cover[local_id]
          uncovered_left_ids[next_uncovered] = lvs_of_component[local_id]
          next_uncovered += 1
        end
      end

      right_cover_m = Vector{Int}(undef, rv_size_of_component)
      @inbounds for (m, local_rv) in enumerate(right_cover)
        right_cover_m[local_rv] = length(left_cover) + m
      end

      @inbounds for lv_id in uncovered_left_ids
        lv = left_vertex(g, lv_id)

        for (rv_id, weight) in weighted_edge_iterator(g, lv_id)
          m = right_cover_m[position_of_rvs_in_component[rv_id]]

          _add_vertex_cover_term!(matrix, m, lv, weight, op_cache, site_dim)
        end
      end
    end

    n == length(sites) && continue

    ## Preallocate space for next_edges
    next_edges = Matrix{Tuple{Vector{Int},Vector{C}}}(
      undef, rank, length(op_cache_vec[n + 1])
    )
    let
      next_edge_sizes = zeros(Int, rank, length(op_cache_vec[n + 1]))
      @inbounds for m in eachindex(left_cover)
        lv_id = lvs_of_component[left_cover[m]]
        for rv_id in g.right_vertex_ids_from_left[lv_id]
          op_id = next_op_of_rv_id[rv_id]
          next_edge_sizes[m, op_id] += 1
        end
      end

      @inbounds for i in eachindex(next_edges)
        n_edges = next_edge_sizes[i]
        next_edges[i] = sizehint!(Int[], n_edges), sizehint!(C[], n_edges)
      end
    end

    ## Construct next_edges for the left_cover
    @inbounds for m in eachindex(left_cover)
      lv_id = lvs_of_component[left_cover[m]]
      for (rv_id, weight) in weighted_edge_iterator(g, lv_id)
        op_id = next_op_of_rv_id[rv_id]

        next_right_vertex_ids, next_edge_weights = next_edges[m, op_id]
        push!(next_right_vertex_ids, rv_id)
        push!(next_edge_weights, weight)
      end
    end

    ## Construct next_edges for the right_cover
    let
      rvs_of_component = Vector{Int}(undef, rv_size_of_component)
      @inbounds for lv_id in lvs_of_component
        right_vertex_ids = g.right_vertex_ids_from_left[lv_id]
        for edge_id in eachindex(right_vertex_ids)
          rv_id = right_vertex_ids[edge_id]
          rvs_of_component[position_of_rvs_in_component[rv_id]] = rv_id
        end
      end

      @inbounds for m in eachindex(right_cover)
        rv_id = rvs_of_component[right_cover[m]]
        m += length(left_cover)

        op_id = next_op_of_rv_id[rv_id]
        next_right_vertex_ids, next_edge_weights = next_edges[m, op_id]
        resize!(next_right_vertex_ids, 1)
        resize!(next_edge_weights, 1)

        next_right_vertex_ids[1] = rv_id
        next_edge_weights[1] = one(C)
      end
    end

    next_edges_of_cc[cc] = next_edges
  end

  return nothing
end
