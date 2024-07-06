BlockSparseMatrix{C} = Dict{Tuple{Int,Int},Matrix{C}}

MPOGraph{N, C} = BipartiteGraph{LeftVertex, NTuple{N, OpID}, C}

@timeit function MPOGraph{N, C}(os::OpIDSum{C}, op_cache_vec::OpCacheVec) where {N, C}
  g = MPOGraph{N, C}([], [], [])

  op_vec = OpID[OpID(0, 0) for _ in 1:N]
  for i in eachindex(os)
    scalar, ops = os[i]

    @assert length(ops) <= N

    for i in 1:length(ops)
      op_vec[i] = ops[end - i + 1]
    end

    for i in (length(ops) + 1):N
      op_vec[i] = OpID(0, 0)
    end

    push!(g.right_vertices, NTuple{N, OpID}(op_vec))
  end

  ## Next we need to sort the right vertices. TODO: Should be able to do this more efficiently I think
  inds = sortperm(g.right_vertices)
  sort!(g.right_vertices)
  weights = os.scalars[inds]

  ## TODO: Break this out into a function that is shared with at_site!
  next_edges = [Vector{Tuple{Int, C}}() for _ in 1:length(op_cache_vec[1])]

  for j in 1:right_size(g)
    weight = weights[j]
    onsite_op = get_onsite_op(right_vertex(g, j), 1)
    push!(next_edges[onsite_op], (j, weight))
  end

  for op_id in 1:length(op_cache_vec[1])
    isempty(next_edges[op_id]) && continue

    first_rv_id = next_edges[op_id][1][1]
    needs_JW_string = is_fermionic(right_vertex(g, first_rv_id), 2, op_cache_vec)
    push!(g.left_vertices, LeftVertex(1, op_id, needs_JW_string))
    push!(g.edges_from_left, next_edges[op_id])
  end

  return g
end


@timeit function sparse_qr(
  A::SparseMatrixCSC, tol::Real
)::Tuple{SparseMatrixCSC,SparseMatrixCSC,Vector{Int},Vector{Int},Int}
  A = sparse(adjoint(A))
  if tol < 0
    ret = qr(A)
  else
    ret = qr(A; tol=tol)
  end

  return sparse(adjoint(ret.R)), sparse(adjoint(ret.Q)), ret.pcol, ret.prow, rank(ret)
end

function for_non_zeros_batch(f::Function, A::SparseMatrixCSC, max_col::Int)::Nothing
  @assert max_col <= size(A, 2) "$max_col, $(size(A, 2))"

  rows = rowvals(A)
  vals = nonzeros(A)
  for col in 1:max_col
    range = nzrange(A, col)
    isempty(range) && continue
    f((@view vals[range]), (@view rows[range]), col)
  end
end

function add_to_local_matrix!(a::Matrix, weight::Number, local_op::Matrix, needs_JW_string::Bool)::Nothing
  if !needs_JW_string
    a .+= weight * local_op
  elseif size(local_op, 1) == 2
    a[:, 1] .+= weight * local_op[:, 1]
    a[:, 2] .-= weight * local_op[:, 2]
  elseif size(local_op, 1) == 4
    a[:, 1] .+= weight * local_op[:, 1]
    a[:, 2] .-= weight * local_op[:, 2]
    a[:, 3] .-= weight * local_op[:, 3]
    a[:, 4] .+= weight * local_op[:, 4]
  else
    error("Unknown fermionic site.")
  end

  return nothing
end

@timeit function at_site!(
  ValType::Type{<:Number},
  g::MPOGraph{N, C},
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  op_cache_vec::OpCacheVec;
  output_level::Int=0)::Tuple{BipartiteGraph,BlockSparseMatrix{ValType},Index} where {N, C}

  has_qns = hasqns(sites)
  matrix = BlockSparseMatrix{ValType}()

  qi = Vector{Pair{QN,Int}}()
  outgoing_link_offset = 0

  next_graph = MPOGraph{N, C}([], g.right_vertices, [])
  
  combine_duplicate_adjacent_right_vertices!(g, terms_eq_from(n + 1))
  ccs = compute_connected_components!(g)

  output_level > 0 && println("  The graph has $(left_size(g)) left vertices, $(right_size(g)) right vertices, $(num_edges(g)) edges and $(num_connected_components(ccs)) connected components")

  for cc in 1:num_connected_components(ccs)
    W, left_map, right_map = get_cc_matrix(g, ccs, cc)

    ## This information is now in W and not needed again.
    clear_edges!(g, left_map)

    ## Compute the decomposition and then free W
    Q, R, prow, pcol, rank = sparse_qr(W, tol)
    W = nothing

    ## If we are at the last site, then Q will be a 1x1 matrix containing an overall phase
    ## that we need to account for.
    if n == length(sites)
      R *= only(Q)
    end

    @timeit "QNs" if has_qns
      right_flux = flux(right_vertex(g, right_map[1]), n + 1, op_cache_vec)
      append!(qi, [QN() - right_flux => rank])
    end

    # Form the local transformation tensor.
    @timeit "R iteration" for_non_zeros_batch(R, length(left_map)) do weights, ms, i
      lv = left_vertex(g, left_map[pcol[i]])
      local_op = op_cache_vec[n][lv.op_id].matrix

      for (weight, m) in zip(weights, ms)
        m > rank && continue
        right_link = m + outgoing_link_offset

        matrix_element = get!(matrix, (lv.link, right_link)) do
          return zeros(C, dim(sites[n]), dim(sites[n]))
        end

        add_to_local_matrix!(matrix_element, weight, local_op, lv.needs_JW_string)
      end
    end

    # Build the graph for the next site. If we are at the last site then
    # we can skip this step.
    @timeit "Q iteration" n != length(sites) && for_non_zeros_batch(Q, rank) do weights, js, m
      next_edges = [Vector{Tuple{Int, C}}() for _ in 1:length(op_cache_vec[n + 1])]

      ## TODO: I think something smart can be done here since we know the right vertices are sorted

      for (weight, j) in zip(weights, js)
        j = right_map[prow[j]]
        onsite_op = get_onsite_op(right_vertex(g, j), n + 1)
        push!(next_edges[onsite_op], (j, weight))
      end

      for op_id in 1:length(op_cache_vec[n + 1])
        isempty(next_edges[op_id]) && continue

        first_rv_id = next_edges[op_id][1][1]
        needs_JW_string = is_fermionic(right_vertex(g, first_rv_id), n + 2, op_cache_vec)
        push!(next_graph.left_vertices, LeftVertex(m + outgoing_link_offset, op_id, needs_JW_string))
        push!(next_graph.edges_from_left, next_edges[op_id])
      end
    end

    outgoing_link_offset += rank
  end

  if has_qns
    outgoing_link = Index(qi...; tags="Link,l=$n", dir=ITensors.Out)
  else
    outgoing_link = Index(outgoing_link_offset; tags="Link,l=$n")
  end

  return next_graph, matrix, outgoing_link
end
