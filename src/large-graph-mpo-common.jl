"""
    BlockSparseMatrix{C}

Vector-backed block-sparse matrix representation used for intermediate MPO
tensor storage.

The outer vector is indexed by component-local `right_link`. Each inner
`Dictionary` maps `left_link` to the dense local operator matrix for that block.
`at_site!` later returns offsets that place component-local right-link ids into
the full outgoing MPO bond.
"""
BlockSparseMatrix{C} = Vector{Dictionary{Int,Matrix{C}}}

"""
    MPOGraph{N,C,Ti}

Type alias for the bipartite graph representation used during MPO construction.

Left vertices store `LeftVertex` metadata for the current site. Right vertices
store fixed-width tuples of `OpID`s describing the remaining operator content of
a term, and edge weights carry scalar coefficients or decomposition weights.
"""
MPOGraph{N,C,Ti} = BipartiteGraph{LeftVertex,NTuple{N,OpID{Ti}},C}

"""
    pretty_print(g::MPOGraph, n::Int, op_cache_vec::OpCacheVec) -> Nothing

Print a human-readable summary of an MPO construction graph at site `n`.

The output includes left-vertex metadata, remaining right-vertex operator
strings after site `n`, and weighted adjacency lists. This is intended for
debugging and writes directly to standard output.
"""
function pretty_print(g::MPOGraph, n::Int, op_cache_vec::OpCacheVec)
  num_left = left_size(g)
  num_right = right_size(g)
  total_edges = num_edges(g)

  println(
    "MPOGraph at site $n has $num_left left vertices, $num_right right vertices and $total_edges edges",
  )
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

`x` is the sortable value and `y` is the carried value.
"""
struct CoSorterElement{T1,T2}
  x::T1
  y::T2
end

"""
    CoSorter{T1,T2,S,C}

Lightweight view that exposes two arrays as a single sortable collection,
ordering by `sortarray` and applying the same swaps to `coarray`.

This is used when sorting `OpIDSum` term storage while keeping the corresponding
scalar coefficients aligned with the same permutation.
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

Right vertices are expected to be ordered so equivalent suffixes are adjacent.
This is used to merge equivalent right vertices after peeling off the operator
acting on the current site.
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
resulting id/weight entry in `next_edges`. `next_edges` must have one row and
one column for each operator available on the next site.
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
`cur_offset` shifts component-local bond indices into the full outgoing link.
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
    MPOGraph(os::OpIDSum{N,C,Ti}; symbolic_coefficients=false) -> MPOGraph{N,C,Ti}

Convert an `OpIDSum` into the initial bipartite graph used by the
MPO construction algorithm.

Operators within each term are put in reverse order (by decreasing site), then
the terms are sorted along with the scalars. This sorting puts terms that share
a terminating sequence of operators (which is now at the front of the term) nearby.
For numeric construction, duplicate terms are then combined and resulting terms
with a weight below `os.abs_tol` are dropped. When `symbolic_coefficients=true`,
scalars are interpreted as signed internal symbolic ids, so duplicate operator
tuples are preserved as separate edge entries instead of being added together.
The returned graph is split before the first site: left vertices represent the
identity incoming link and the operator emitted at site 1, while right vertices
carry the remaining operator tuples.
"""
@timeit function MPOGraph(
  os::OpIDSum{N,C,Ti}; symbolic_coefficients::Bool=false
)::MPOGraph{N,C,Ti} where {N,C,Ti}
  @assert size(os.terms, 1) == N
  if symbolic_coefficients && !(C <: Integer)
    throw(ArgumentError("Symbolic MPO graph construction requires integer coefficient ids."))
  end

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

  ## Combine duplicate numeric terms and remove terms which are below the tolerance.
  ## Symbolic ids are labels, not numeric weights, so preserve duplicate entries.
  nnz = 0
  if symbolic_coefficients
    for i in eachindex(os)
      _check_symbolic_weight_id(os.scalars[i])
      nnz += 1
      os.scalars[nnz] = os.scalars[i]
      os._data[nnz] = os._data[i]
    end
  else
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
    add_to_local_matrix!(a, weight, local_op, needs_JW_string) -> Nothing

Accumulate a weighted local operator matrix into `a`.

If `needs_JW_string` is `true`, the contribution is multiplied by the diagonal
Jordan-Wigner sign pattern expected for 2-state or 4-state fermionic sites.
`a` is mutated in place and `local_op` is read without copying.
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

The first returned vector is the component order to use when laying out the
outgoing bond space, and the second is the merged QN-sector description used to
construct the outgoing link `Index`.
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

function _check_qr_block_storage(::Type{SymbolicBlockSparseMatrix{Ti}})::Nothing where {Ti}
  throw(ArgumentError("QR construction is only supported for numeric block storage."))
end

function _check_qr_block_storage(::Type)::Nothing
  return nothing
end

"""
    at_site!(ValType, g, n, sites, tol, absolute_tol, op_cache_vec, alg;
             combine_qn_sectors, output_level=0)
        -> Tuple{MPOGraph,Vector{Int},Vector{BlockSparseMatrix{ValType}},Index}

Process one site of the MPO construction algorithm.

Equivalent adjacent right vertices are compacted, connected components are
computed, and each component is processed by either the sparse-QR path
(`alg == "QR"`) or the minimum-vertex-cover path (`alg == "VC"`). The
returned tuple contains:
- the graph for site `n + 1`,
- offsets locating each connected component inside the outgoing bond space,
- the block-sparse local MPO tensors for each component,
- the outgoing link `Index`, optionally grouped into merged QN sectors.

`tol` and `absolute_tol` are used only by the QR path. `combine_qn_sectors`
merges adjacent outgoing link sectors with the same QN after component ranks are
known.
"""
@timeit function at_site!(
  ::Type{MatrixType},
  g::MPOGraph{N,C,Ti},
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  absolute_tol::Bool,
  op_cache_vec::OpCacheVec,
  alg::String;
  combine_qn_sectors::Bool,
  output_level::Int=0,
)::Tuple{MPOGraph{N,C,Ti},Vector{Int},Vector{MatrixType},Index} where {MatrixType,N,C,Ti}
  has_qns = hasqns(sites)

  workspace = combine_duplicate_adjacent_right_vertices!(g, terms_eq_from(n + 1))

  ccs = compute_connected_components(g, workspace)
  nccs = num_connected_components(ccs)
  workspace = nothing

  ## The rank of each connected component.
  rank_of_cc = zeros(Int, nccs)

  ## The MPO tensor for each component.
  matrix_of_cc = [_vertex_cover_matrix(MatrixType, 0) for _ in 1:nccs]

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
    _check_qr_block_storage(MatrixType)
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
      matrix_of_cc, rank_of_cc, next_edges_of_cc, g, ccs, n, sites, op_cache_vec
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
    output_level > 1 &&
      println("  Total rank is $cur_offset with $(length(qi_of_cc)) different QN sectors.")
  else
    outgoing_link = Index(cur_offset; tags="Link,l=$n")
    output_level > 1 && println("    Total rank is $cur_offset.")
  end

  return next_graph, offset_of_cc, matrix_of_cc, outgoing_link
end
