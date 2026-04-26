"""
    sparse_qr(A::SparseMatrixCSC, tol::Real, absolute_tol::Bool)
        -> Tuple{Q,R,prow,pcol,rank}

Compute a sparse QR factorization of `A` using SuiteSparse SPQR.

If `absolute_tol` is `false`, `tol` is interpreted relative to SPQR's default
tolerance scale. The return values are the orthogonal factor `Q`, upper-triangular
factor `R`, row and column permutations, and the numerical rank. SPQR uses the
current BLAS thread count for its internal thread setting.
"""
function sparse_qr(
  A::SparseMatrixCSC, tol::Real, absolute_tol::Bool
)::Tuple{SparseArrays.SPQR.QRSparseQ,SparseMatrixCSC,Vector{Int},Vector{Int},Int}
  ret = nothing

  ## The tolerance is specified in Section 2.3
  ## https://fossies.org/linux/SuiteSparse/SPQR/Doc/spqr_user_guide.pdf
  if !absolute_tol
    tol *= SparseArrays.SPQR._default_tol(A)
  end

  SparseArrays.CHOLMOD.@cholmod_param SPQR_nthreads = BLAS.get_num_threads() begin
    ret = qr(A; tol)
  end

  return ret.Q, ret.R, ret.prow, ret.pcol, rank(ret)
end

"""
    for_non_zeros_batch(f::Function, A::SparseMatrixCSC, max_col::Int) -> Nothing

Iterate over the nonzero entries of the first `max_col` columns of `A`, calling
`f(values, rows, col)` once per nonempty column.

`values` and `rows` are views into the CSC storage of `A` for that column, so
the callback must not retain them beyond the call.
"""
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

"""
    for_non_zeros_batch(f::Function, Q::SparseArrays.SPQR.QRSparseQ, max_col::Int) -> Nothing

Iterate over the first `max_col` columns of a sparse SPQR `Q` factor, forming
each column explicitly and calling `f(column, col)`. The dense column workspace
is reused between calls, so the callback must copy it if it needs to keep it.
"""
function for_non_zeros_batch(
  f::Function, Q::SparseArrays.SPQR.QRSparseQ, max_col::Int
)::Nothing
  @assert max_col <= size(Q, 2) "$max_col, $(size(Q, 2))"

  function get_column!(Q::SparseArrays.SPQR.QRSparseQ, col::Int, res::Vector)
    res .= 0
    res[col] = 1

    for l in size(Q.factors, 2):-1:1
      τl = -Q.τ[l]
      h = view(Q.factors, :, l)
      axpy!(τl*dot(h, res), h, res)
    end

    return res
  end

  res = zeros(eltype(Q), size(Q, 1))
  for col in 1:max_col
    get_column!(Q, col, res)
    f(res, col)
  end
end

"""
    process_qr(
      matrix_of_cc, rank_of_cc, next_edges_of_cc, g, ccs, n, sites, tol, absolute_tol, op_cache_vec
    ) -> Nothing

Process every connected component using the sparse-QR path.

Single-left-vertex components use `process_single_left_vertex_cc!`; larger
components are extracted as sparse weighted adjacency matrices, decomposed with
`sparse_qr`, and translated into local tensor blocks plus outgoing graph edges.
On the final site, the scalar `R` factor is applied directly to the local
blocks.
"""
@timeit function process_qr(
  matrix_of_cc::Vector{BlockSparseMatrix{ValType}},
  rank_of_cc::Vector{Int},
  next_edges_of_cc::Vector{Matrix{Tuple{Vector{Int},Vector{C}}}},
  g::MPOGraph{N,C,Ti},
  ccs::BipartiteGraphConnectedComponents,
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  absolute_tol::Bool,
  op_cache_vec::OpCacheVec,
)::Nothing where {ValType<:Number,N,C,Ti}
  Threads.@threads for cc in 1:num_connected_components(ccs)
    ## A specialization for when there is only one vertex on the left. This is
    ## a very common case that can be sped up significantly.
    if left_size(ccs, cc) == 1
      process_single_left_vertex_cc!(
        matrix_of_cc, rank_of_cc, next_edges_of_cc, g, ccs, cc, n, sites, op_cache_vec
      )
      continue
    end

    matrix = matrix_of_cc[cc]
    W, left_map, right_map = get_cc_matrix(g, ccs, cc; clear_edges=true)

    ## Compute the decomposition and then free W
    Q, R, prow, pcol, rank = sparse_qr(W, tol, absolute_tol)
    W = nothing

    rank_of_cc[cc] = rank

    ## Form the local transformation tensor.
    for_non_zeros_batch(Q, rank) do weights::AbstractVector{C}, m::Int
      for (i, weight) in enumerate(weights)
        weight == 0 && continue

        ## TODO: This allocates for some reason
        lv = left_vertex(g, left_map[prow[i]])
        local_op = op_cache_vec[n][lv.op_id].matrix

        matrix_element = get!(matrix, (lv.link, m)) do
          return zeros(ValType, dim(sites[n]), dim(sites[n]))
        end

        add_to_local_matrix!(matrix_element, weight, local_op, lv.needs_JW_string)
      end
    end

    ## Q and prow are no longer needed.
    Q = nothing
    prow = nothing

    ## If we are at the last site, then R will be a 1x1 matrix containing an overall scaling.
    if n == length(sites)
      scaling = only(R)
      for block in values(matrix)
        block .*= scaling
      end

      continue
    end

    next_edges_size = zeros(Int, rank, length(op_cache_vec[n + 1]))
    rv_id_lookup = Vector{Int}(undef, size(R, 2))
    op_id_lookup = Vector{Ti}(undef, size(R, 2))

    for_non_zeros_batch(
      R, length(right_map)
    ) do _::AbstractVector{C}, ms::AbstractVector{Int}, j::Int
      ## Convert j, which has been permuted first by the connected components
      ## and then again by SPQR into a right vertex Id.
      rv_id = right_map[pcol[j]]

      ## Get the operator acting on site (n + 1) of this right vertex.
      op_id = get_onsite_op(right_vertex(g, rv_id), n + 1)

      rv_id_lookup[j] = rv_id
      op_id_lookup[j] = op_id

      for m in ms
        next_edges_size[m, op_id] += 1
      end
    end

    ## Build the graph for the next site out of this component.
    next_edges = Matrix{Tuple{Vector{Int},Vector{C}}}(
      undef, rank, length(op_cache_vec[n + 1])
    )
    for i in eachindex(next_edges)
      right_vertex_ids = Int[]
      edge_weights = C[]
      sizehint!(right_vertex_ids, next_edges_size[i])
      sizehint!(edge_weights, next_edges_size[i])
      next_edges[i] = (right_vertex_ids, edge_weights)
    end

    for_non_zeros_batch(
      R, length(right_map)
    ) do weights::AbstractVector{C}, ms::AbstractVector{Int}, j::Int
      rv_id = rv_id_lookup[j]
      op_id = op_id_lookup[j]

      ## Add the edges.
      for (m, weight) in zip(ms, weights)
        next_right_vertex_ids, next_edge_weights = next_edges[m, op_id]
        push!(next_right_vertex_ids, rv_id)
        push!(next_edge_weights, weight)
      end
    end

    next_edges_of_cc[cc] = next_edges
  end

  return nothing
end

