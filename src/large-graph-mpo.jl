"""
    BlockSparseMatrix{C}

Dictionary-backed block-sparse matrix representation used for intermediate MPO tensor storage.

Keys are `(left_link, right_link)` pairs and values are dense local operator
matrices for that block.
"""
BlockSparseMatrix{C} = Dict{Tuple{Int,Int},Matrix{C}}
# TODO: consider changing to Vector{Dict{Int, Matrix{C}}} from right_link => (left_link => matrix) and use Dictionaries.jl

"""
    MPOGraph{N,C,Ti}

Type alias for the bipartite graph representation used during MPO construction.

Left vertices store `LeftVertex` metadata, right vertices store fixed-width tuples of `OpID`s describing the
remaining operator content of a term, and edge weights carry the scalar coefficients.
"""
MPOGraph{N,C,Ti} = BipartiteGraph{LeftVertex,NTuple{N,OpID{Ti}},C}

function pretty_print(g::MPOGraph, n::Int, op_cache_vec::OpCacheVec)
  num_left = left_size(g)
  num_right = right_size(g)
  total_edges = num_edges(g)

  println("MPOGraph at site $n has $num_left left vertices, $num_right right vertices and $total_edges edges")
  println("  Left vertices:")
  for (lv_id, lv) in enumerate(g.left_vertices)
    op = op_cache_vec[n][lv.op_id].name
    println("    $lv_id: link = $(lv.link), op = $op, fermionic = $(lv.needs_JW_string)")
  end

  println("  Right vertices:")
  for (rv_id, rv) in enumerate(g.right_vertices)
    print("    $rv_id:")

    is_identity = true
    for op in reverse(rv)
      op.n <= n && continue
      name = op_cache_vec[op.n][op.id].name
      print(" $name^$(op.n)")
      is_identity = false
    end

    is_identity && print(" I")
    println()
  end

  println("  Edges from left:")
  for lv_id in 1:left_size(g)
    print("    $lv_id: ")
    for (rv_id, weight) in weighted_edge_iterator(g, lv_id)
      print("($rv_id, $weight), ")
    end
    println()
  end
end

"""
    CoSorterElement{T1,T2}

Pair-like element used by `CoSorter` so that sorting one array carries along a
second array with the same permutation.

Taken from https://discourse.julialang.org/t/how-to-sort-two-or-more-lists-at-once/12073/13
"""
struct CoSorterElement{T1,T2}
  x::T1
  y::T2
end

"""
    CoSorter{T1,T2,S,C}

Lightweight view that exposes two arrays as a single sortable collection,
ordering by `sortarray` and applying the same swaps to `coarray`.
"""
struct CoSorter{T1,T2,S<:AbstractArray{T1},C<:AbstractArray{T2}} <:
       AbstractVector{CoSorterElement{T1,T2}}
  sortarray::S
  coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)

function Base.getindex(c::CoSorter, i...)
  CoSorterElement(getindex(c.sortarray, i...), getindex(c.coarray, i...))
end

function Base.setindex!(c::CoSorter, t::CoSorterElement, i...)
  (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c)
end

Base.isless(a::CoSorterElement, b::CoSorterElement) = isless(a.x, b.x)

"""
    find_first_eq_rv(g::MPOGraph, j::Int, n::Int) -> Int

Walk backward from right-vertex index `j` to the first earlier right vertex
whose operator tuple is equivalent from site `n` onward.

This is used to "merge" equivalent right vertices after peeling off the
operator acting on the current site.
"""
function find_first_eq_rv(g::MPOGraph, j::Int, n::Int)::Int
  while j > 1 && are_equal(right_vertex(g, j), right_vertex(g, j - 1), n)
    j -= 1
  end

  return j
end

"""
    build_next_edges_specialization!(next_edges, g, cur_site, right_vertex_ids, edge_weights) -> Nothing

Fast path for building outgoing edges when a connected component has only
a single left vertex.

For each current edge, this extracts the operator acting on `cur_site + 1`,
finds the unique right vertex from `cur_site + 2` onward, and stores the
resulting id/weight entry in `next_edges`.
"""
function build_next_edges_specialization!(
  next_edges::Matrix{Tuple{Vector{Int},Vector{C}}},
  g::MPOGraph{N,C,Ti},
  cur_site::Int,
  right_vertex_ids,
  edge_weights,
)::Nothing where {N,C,Ti}
  @assert size(next_edges, 1) == 1
  @assert length(right_vertex_ids) == length(edge_weights)

  for edge_id in eachindex(right_vertex_ids)
    rv_id = right_vertex_ids[edge_id]
    weight = edge_weights[edge_id]
    op_id = get_onsite_op(right_vertex(g, rv_id), cur_site + 1)
    next_right_vertex_ids, next_edge_weights = next_edges[1, op_id]

    push!(next_right_vertex_ids, rv_id)
    push!(next_edge_weights, weight)
  end

  return nothing
end

"""
    add_to_next_graph!(next_graph, cur_graph, op_cache_vec, cur_site, cur_offset, next_edges) -> Nothing

Append the left vertices and adjacency lists described by `next_edges` to `next_graph`.

Each nonempty `(bond_index, op_id)` entry in `next_edges` creates one `LeftVertex`, reusing that
entry's `(right_vertex_ids, edge_weights)` vectors as the outgoing adjacency list. The stored
`needs_JW_string` flag is inferred from the fermionic parity of the connected right vertices.
"""
function add_to_next_graph!(
  next_graph::MPOGraph{N,C,Ti},
  cur_graph::MPOGraph{N,C,Ti},
  op_cache_vec::OpCacheVec,
  cur_site::Int,
  cur_offset::Int,
  next_edges::Matrix{Tuple{Vector{Int},Vector{C}}},
)::Nothing where {N,C,Ti}
  for op_id in 1:size(next_edges, 2)
    for m in 1:size(next_edges, 1)
      cur_right_vertex_ids, cur_edge_weights = next_edges[m, op_id]
      isempty(cur_right_vertex_ids) && continue

      first_rv_id = cur_right_vertex_ids[1]
      needs_JW_string = is_fermionic(
        right_vertex(cur_graph, first_rv_id), cur_site + 2, op_cache_vec
      )
      push!(next_graph.left_vertices, LeftVertex(m + cur_offset, op_id, needs_JW_string))
      push!(next_graph.right_vertex_ids_from_left, cur_right_vertex_ids)
      push!(next_graph.edge_weights_from_left, cur_edge_weights)
    end
  end

  return nothing
end

"""
    MPOGraph(os::OpIDSum{N,C,Ti}) -> MPOGraph{N,C,Ti}

Convert an `OpIDSum` into the initial bipartite graph used by the
MPO construction algorithm.

Operators with each term are put in reverse order (by decreasing site), then
the terms are sorted along with the scalars. This sorting puts terms which share
a terminating sequence of operators (which is now at the front of the term) nearby.
Duplicate terms are then combined, and resulting terms with a weight below
the `os.abs_tol` are dropped. The returned graph is split about the first site.
"""
@timeit function MPOGraph(os::OpIDSum{N,C,Ti})::MPOGraph{N,C,Ti} where {N,C,Ti}
  @assert size(os.terms, 1) == N

  ## Reverse the terms in the sum, ignoring trailing identity operators.
  Threads.@threads for i in 1:length(os)
    for j in N:-1:1
      if os.terms[j, i] != zero(os.terms[j, i])
        reverse!(view(os.terms, 1:j, i))
        break
      end
    end
  end

  ## Sort the terms and scalars.
  resize!(os._data, length(os))
  resize!(os.scalars, length(os))
  sort!(
    CoSorter(os._data, os.scalars);
    alg=(Threads.nthreads() > 1) ? ThreadsX.QuickSort : Base.QuickSort,
  )

  ## Combine duplicate terms and remove terms which are below the tolerance.
  nnz = 0
  for i in eachindex(os)
    if i < length(os) && os._data[i] == os._data[i + 1]
      os.scalars[i + 1] += os.scalars[i]
      os.scalars[i] = 0
    elseif abs(os.scalars[i]) > os.abs_tol
      nnz += 1
      os.scalars[nnz] = os.scalars[i]
      os._data[nnz] = os._data[i]
    end
  end

  os.num_terms[] = nnz
  resize!(os._data, nnz)
  resize!(os.scalars, nnz)

  g = MPOGraph{N,C,Ti}([], os._data, [], [])

  next_edges = Matrix{Tuple{Vector{Int},Vector{C}}}(undef, 1, length(os.op_cache_vec[1]))
  for i in eachindex(next_edges)
    next_edges[i] = (Int[], C[])
  end

  build_next_edges_specialization!(
    next_edges, g, 0, Base.OneTo(length(os.scalars)), os.scalars
  )

  add_to_next_graph!(g, g, os.op_cache_vec, 0, 0, next_edges)

  return g
end

"""
    sparse_qr(A::SparseMatrixCSC, tol::Real, absolute_tol::Bool)
        -> Tuple{Q,R,prow,pcol,rank}

Compute a sparse QR factorization of `A` using SuiteSparse SPQR.

If `absolute_tol` is `false`, `tol` is interpreted relative to SPQR's default
tolerance scale. The return values are the orthogonal factor `Q`, upper-triangular
factor `R`, row and column permutations, and the numerical rank.
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

`values` and `rows` are views into the storage of `A` for that column.
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
each column explicitly and calling `f(column, col)`.
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
    add_to_local_matrix!(a, weight, local_op, needs_JW_string) -> Nothing

Accumulate a weighted local operator matrix into `a`.

If `needs_JW_string` is `true`, the contribution is multiplied by the diagonal
Jordan-Wigner sign pattern expected for 2-state or 4-state fermionic sites.
"""
function add_to_local_matrix!(
  a::Matrix, weight::Number, local_op::Matrix, needs_JW_string::Bool
)::Nothing
  if !needs_JW_string
    @inbounds for i in CartesianIndices(a)
      a[i] += weight * local_op[i]
    end
  elseif size(local_op, 1) == 2
    @inbounds for i in 1:2
      a[i, 1] += weight * local_op[i, 1]
      a[i, 2] -= weight * local_op[i, 2]
    end
  elseif size(local_op, 1) == 4
    @inbounds for i in 1:4
      a[i, 1] += weight * local_op[i, 1]
      a[i, 2] -= weight * local_op[i, 2]
      a[i, 3] -= weight * local_op[i, 3]
      a[i, 4] += weight * local_op[i, 4]
    end
  else
    error("Unknown fermionic site.")
  end

  return nothing
end

"""
    merge_qn_sectors(qi_of_cc::Vector{Pair{QN,Int}})
        -> Tuple{Vector{Int},Vector{Pair{QN,Int}}}

Sort connected components by QN sector and merge adjacent entries with the same
QN by summing their dimensions.

The first returned vector is the permutation that reorders components, and the
second is the merged QN-sector description.
"""
function merge_qn_sectors(
  qi_of_cc::Vector{Pair{QN,Int}}
)::Tuple{Vector{Int},Vector{Pair{QN,Int}}}
  new_order = sortperm(qi_of_cc; by=pair -> pair[1])
  qi_of_cc = sort(qi_of_cc; by=pair -> pair[1])

  new_qi = Pair{QN,Int}[qi_of_cc[1]]
  for qi in view(qi_of_cc, 2:length(qi_of_cc))
    if qi.first == new_qi[end].first
      new_qi[end] = qi.first => new_qi[end].second + qi.second
    else
      push!(new_qi, qi)
    end
  end

  return new_order, new_qi
end

"""
    process_single_left_vertex_cc!(
      matrix_of_cc, qi_of_cc, rank_of_cc, next_edges_of_cc, g, ccs, cc, n, sites, op_cache_vec
    ) -> Nothing

Handle the common connected-component case with exactly one left vertex.

This fills the local MPO tensor contribution for the component, records its QN
sector and rank, and either applies the terminal scaling on the last site or
builds the outgoing edges for the next site.
"""
function process_single_left_vertex_cc!(
  matrix_of_cc::Vector{BlockSparseMatrix{ValType}},
  rank_of_cc::Vector{Int},
  next_edges_of_cc::Vector{Matrix{Tuple{Vector{Int},Vector{C}}}},
  g::MPOGraph{N,C,Ti},
  ccs::BipartiteGraphConnectedComponents,
  cc::Int,
  n::Int,
  sites::Vector{<:Index},
  op_cache_vec::OpCacheVec,
)::Nothing where {ValType<:Number,N,C,Ti}
  lv_id = only(ccs.lvs_of_component[cc])
  rank = 1
  rank_of_cc[cc] = rank

  lv = left_vertex(g, lv_id)
  local_op = op_cache_vec[n][lv.op_id].matrix

  matrix_element = get!(matrix_of_cc[cc], (lv.link, rank)) do
    return zeros(ValType, dim(sites[n]), dim(sites[n]))
  end

  add_to_local_matrix!(matrix_element, one(ValType), local_op, lv.needs_JW_string)

  if n == length(sites)
    scaling = only(g.edge_weights_from_left[lv_id])

    for block in values(matrix_of_cc[cc])
      block .*= scaling
    end

    return nothing
  end

  next_edges = Matrix{Tuple{Vector{Int},Vector{C}}}(
    undef, rank, length(op_cache_vec[n + 1])
  )
  for i in eachindex(next_edges)
    next_edges[i] = (Int[], C[])
  end

  build_next_edges_specialization!(
    next_edges, g, n, g.right_vertex_ids_from_left[lv_id], g.edge_weights_from_left[lv_id]
  )

  clear_edges_from_left!(g, lv_id)

  next_edges_of_cc[cc] = next_edges

  return nothing
end

"""
    process_vertex_cover!(
      matrix_of_cc, rank_of_cc, next_edges_of_cc, g, ccs, n, sites, op_cache_vec
    ) -> Nothing

Process every connected component using the minimum-vertex-cover specialization.
"""
@timeit function process_vertex_cover!(
  matrix_of_cc::Vector{BlockSparseMatrix{ValType}},
  rank_of_cc::Vector{Int},
  next_edges_of_cc::Vector{Matrix{Tuple{Vector{Int},Vector{C}}}},
  g::MPOGraph{N,C,Ti},
  ccs::BipartiteGraphConnectedComponents,
  n::Int,
  sites::Vector{<:Index},
  op_cache_vec::OpCacheVec,
)::Nothing where {ValType<:Number,N,C,Ti}
  site_dim = dim(sites[n])
  op_cache = op_cache_vec[n]

  next_op_of_rv_id = Ti[]
  if n != length(sites)
    resize!(next_op_of_rv_id, right_size(g))
    Threads.@threads for rv_id in 1:right_size(g)
      next_op_of_rv_id[rv_id] = get_onsite_op(right_vertex(g, rv_id), n + 1)
    end
  end

  # TODO: Consider nested multithreading
  Threads.@threads for cc in 1:num_connected_components(ccs)
    matrix = matrix_of_cc[cc]
    lvs_of_component::Vector{Int} = ccs.lvs_of_component[cc]
    position_of_rvs_in_component = ccs.position_of_rvs_in_component
    rv_size_of_component = ccs.rv_size_of_component[cc]

    ## No idea why, but these need to be typed or allocations go nuts.
    left_cover::Vector{Int}, right_cover::Vector{Int} = minimum_vertex_cover(g, ccs, cc)

    rank = length(left_cover) + length(right_cover)
    rank_of_cc[cc] = rank

    ## Construct the tensor from the left cover.
    @inbounds for m in eachindex(left_cover)
      lv_id = lvs_of_component[left_cover[m]]
      lv = left_vertex(g, lv_id)
      local_op = op_cache[lv.op_id].matrix

      matrix_element = zeros(ValType, site_dim, site_dim)
      add_to_local_matrix!(matrix_element, one(ValType), local_op, lv.needs_JW_string)
      matrix[lv.link, m] = matrix_element
    end

    ## Construct the tensor from the right cover.
    # TODO: Merge this loop with the one above, using `in_left_cover`
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
        local_op = op_cache[lv.op_id].matrix

        for (rv_id, weight) in weighted_edge_iterator(g, lv_id)
          m = right_cover_m[position_of_rvs_in_component[rv_id]]

          matrix_element = get!(matrix, (lv.link, m)) do
            zeros(ValType, site_dim, site_dim)
          end

          add_to_local_matrix!(matrix_element, weight, local_op, lv.needs_JW_string)
        end
      end
    end

    n == length(sites) && continue

    ## Preallocate space for next_edges
    next_edges = Matrix{Tuple{Vector{Int},Vector{C}}}(undef, rank, length(op_cache_vec[n + 1]))
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

"""
    process_qr(
      matrix_of_cc, rank_of_cc, next_edges_of_cc, g, ccs, n, sites, tol, absolute_tol, op_cache_vec
    ) -> Nothing

Process every connected component using the sparse-QR path.
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
        matrix_of_cc,
        rank_of_cc,
        next_edges_of_cc,
        g,
        ccs,
        cc,
        n,
        sites,
        op_cache_vec,
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
    next_edges = Matrix{Tuple{Vector{Int},Vector{C}}}(undef, rank, length(op_cache_vec[n + 1]))
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

"""
    at_site!(ValType, g, n, sites, tol, absolute_tol, op_cache_vec; combine_qn_sectors, output_level=0)
        -> Tuple{MPOGraph,Vector{Int},Vector{BlockSparseMatrix{ValType}},Index}

Process one site of the MPO construction algorithm.

For each connected component of `g`, this routine dispatches to either the
minimum-vertex-cover or sparse-QR processing path, assembles the local MPO
tensor blocks for site `n`, and builds the graph passed to the next site. The
returned tuple contains:
- the graph for site `n + 1`,
- offsets locating each connected component inside the outgoing bond space,
- the block-sparse local MPO tensors for each component,
- the outgoing link `Index`, optionally grouped into merged QN sectors.
"""
@timeit function at_site!(
  ::Type{ValType},
  g::MPOGraph{N,C,Ti},
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  absolute_tol::Bool,
  op_cache_vec::OpCacheVec,
  alg::String;
  combine_qn_sectors::Bool,
  output_level::Int=0,
)::Tuple{
  MPOGraph{N,C,Ti},Vector{Int},Vector{BlockSparseMatrix{ValType}},Index
} where {ValType<:Number,N,C,Ti}
  has_qns = hasqns(sites)

  workspace = combine_duplicate_adjacent_right_vertices!(g, terms_eq_from(n + 1))

  ccs = compute_connected_components(g, workspace)
  nccs = num_connected_components(ccs)
  workspace = nothing

  ## The rank of each connected component.
  rank_of_cc = zeros(Int, nccs)

  ## The MPO tensor for each component.
  matrix_of_cc = [BlockSparseMatrix{ValType}() for _ in 1:nccs]

  ## The QN of each component
  qi_of_cc = Vector{Pair{QN,Int}}(undef, nccs)
  has_qns && for cc in 1:nccs
    first_lv_id = ccs.lvs_of_component[cc][1]
    first_rv_id = g.right_vertex_ids_from_left[first_lv_id][1]
    qn = flux(right_vertex(g, first_rv_id), n + 1, op_cache_vec)
    qi_of_cc[cc] = qn => -1
  end

  ## A map from the incoming link to the next site (outgoing link from this site) and the
  ## operator on the next site (this uniquely specifies the left vertex of the next site)
  ## to the right vertices it will connect to along with the weight.
  next_edges_of_cc = [Matrix{Tuple{Vector{Int},Vector{C}}}(undef, 0, 0) for _ in 1:nccs]

  output_level > 1 && println(
    "  The graph is $(left_size(g)) × $(right_size(g)) with $(num_edges(g)) edges and $(nccs) connected components.",
  )

  if alg == "QR"
    process_qr(
      matrix_of_cc,
      rank_of_cc,
      next_edges_of_cc,
      g,
      ccs,
      n,
      sites,
      tol,
      absolute_tol,
      op_cache_vec,
    )
  elseif alg == "VC"
    process_vertex_cover!(
      matrix_of_cc,
      rank_of_cc,
      next_edges_of_cc,
      g,
      ccs,
      n,
      sites,
      op_cache_vec,
    )
  else
    throw(ArgumentError("The supported algorithms are 'QR' and 'VC': $alg"))
  end

  for cc in 1:nccs
    qn = first(qi_of_cc[cc])
    qi_of_cc[cc] = qn => rank_of_cc[cc]
  end

  cc_order = [i for i in 1:nccs]
  if combine_qn_sectors && has_qns
    cc_order, qi_of_cc = merge_qn_sectors(qi_of_cc)
  end

  ## Combine the graphs of each component together
  next_graph = MPOGraph{N,C,Ti}([], g.right_vertices, [], [])
  offset_of_cc = zeros(Int, nccs)

  cur_offset = 0
  for cc in cc_order
    offset_of_cc[cc] = cur_offset
    add_to_next_graph!(next_graph, g, op_cache_vec, n, cur_offset, next_edges_of_cc[cc])
    cur_offset += rank_of_cc[cc]
  end

  if has_qns
    outgoing_link = Index(qi_of_cc; tags="Link,l=$n", dir=ITensors.Out)
    output_level > 1 && println(
      "  Total rank is $cur_offset with $(length(qi_of_cc)) different QN sectors."
    )
  else
    outgoing_link = Index(cur_offset; tags="Link,l=$n")
    output_level > 1 && println("    Total rank is $cur_offset.")
  end

  return next_graph, offset_of_cc, matrix_of_cc, outgoing_link
end
