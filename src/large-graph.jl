"""
    BipartiteGraph{L,R,C}

Weighted bipartite graph with typed left and right vertices.

Type Parameters
- `L`: The type of the left vertices.
- `R`: The type of the right vertices.
- `C`: The scalar edge weight type.

Fields:
- `left_vertices`: metadata stored for each left vertex.
- `right_vertices`: metadata stored for each right vertex.
- `edges_from_left`: adjacency list from each left vertex to right vertices,
  stored as `(right_vertex_id, weight)` pairs.
"""
struct BipartiteGraph{L,R,C}
  left_vertices::Vector{L}
  right_vertices::Vector{R}
  edges_from_left::Vector{Vector{Tuple{Int,C}}}
end

"""
    left_size(g::BipartiteGraph) -> Int

Return the number of left vertices in `g`.
"""
left_size(g::BipartiteGraph)::Int = length(g.left_vertices)

"""
    right_size(g::BipartiteGraph) -> Int

Return the number of right vertices in `g`.
"""
right_size(g::BipartiteGraph)::Int = length(g.right_vertices)

"""
    left_vertex(g::BipartiteGraph, lv_id::Integer)

Return the data associated with the left vertex `lv_id`.
"""
left_vertex(g::BipartiteGraph, lv_id::Integer) = g.left_vertices[lv_id]

"""
    right_vertex(g::BipartiteGraph, rv_id::Integer)

Return the data associated with the right vertex `rv_id`.
"""
right_vertex(g::BipartiteGraph, rv_id::Integer) = g.right_vertices[rv_id]

"""
    num_edges(g::BipartiteGraph) -> Int

Return the total number of edges in `g`.
"""
function num_edges(g::BipartiteGraph)::Int
  return sum(length(rvs) for rvs in g.edges_from_left)
end

"""
    BipartiteGraphConnectedComponents

Struct containing the connected components of a `BipartiteGraph`.

Fields:
- `lvs_of_component`: for each connected component, the global ids of its left
  vertices.
- `component_position_of_rvs`: for each global right vertex id, its local id
  (position) within its connected component, or an unused sentinel for isolated
  right vertices.
- `rv_size_of_component`: number of right vertices in each component.
"""
struct BipartiteGraphConnectedComponents
  lvs_of_component::Vector{Vector{Int}}
  component_position_of_rvs::Vector{Int}
  rv_size_of_component::Vector{Int}
end

"""
    compute_connected_components(g::BipartiteGraph) -> BipartiteGraphConnectedComponents

Compute the connected components of `g`, keeping only components which connect
the left and right sides of the bipartite graph (i.e. discarding isolated vertices).

The returned object records, for each retained component, which left vertices it
contains and how global right-vertex ids map into local positions within their component.
"""
function compute_connected_components(
  g::BipartiteGraph
)::BipartiteGraphConnectedComponents
  return compute_connected_components(g, Vector{Int}(undef, right_size(g)))
end

@timeit function compute_connected_components(
  g::BipartiteGraph, workspace::Vector{Int}
)::BipartiteGraphConnectedComponents
  @assert length(workspace) >= right_size(g)
  resize!(workspace, right_size(g))
  min_lv_connected_to_rv = workspace
  min_lv_connected_to_rv .= typemax(Int)

  component_of_lv = Int[i for i in 1:left_size(g)]
  lvs_of_component = Vector{Int}[[i] for i in 1:left_size(g)]

  for lv_id in 1:left_size(g)
    cur_min_lv = lv_id

    for (rv_id, _) in g.edges_from_left[lv_id]
      min_lv_of_rv = min_lv_connected_to_rv[rv_id]

      ## If the right vertex has not yet been reached...
      if min_lv_of_rv == typemax(Int)
        min_lv_connected_to_rv[rv_id] = lv_id
        continue
      end

      ## Otherwise, merge the two components
      cur_min_lv, src_lv = minmax(cur_min_lv, min_lv_of_rv)
      cur_component = component_of_lv[cur_min_lv]
      src_component = component_of_lv[src_lv]

      cur_component == src_component && continue

      for lv_id in lvs_of_component[src_component]
        component_of_lv[lv_id] = cur_component
      end

      append!(lvs_of_component[cur_component], lvs_of_component[src_component])
      empty!(lvs_of_component[src_component])
      sizehint!(lvs_of_component[src_component], 0)
    end
  end

  ## Mutate min_lv_connected_to_rv which stores the first left vertex connected to each
  ## right vertex into the position of each right vertex within it's component.
  rv_size_of_component = zeros(Int, length(lvs_of_component))
  for rv_id in 1:right_size(g)
    min_lv = min_lv_connected_to_rv[rv_id]

    ## This means the right vertex is not connected to anything and can be safely ignored.
    min_lv == typemax(Int) && continue

    component = component_of_lv[min_lv]
    rv_size_of_component[component] += 1
    min_lv_connected_to_rv[rv_id] = rv_size_of_component[component]
  end

  ## Drop empty components.
  lvs_of_component_non_empty = Vector{Vector{Int}}()
  rv_size_of_component_non_empty = Vector{Int}()
  for (i, lvs) in enumerate(lvs_of_component)
    if !isempty(lvs) && rv_size_of_component[i] != 0
      push!(lvs_of_component_non_empty, lvs)
      push!(rv_size_of_component_non_empty, rv_size_of_component[i])
    end
  end

  return BipartiteGraphConnectedComponents(
    lvs_of_component_non_empty, min_lv_connected_to_rv, rv_size_of_component_non_empty
  )
end

"""
    num_connected_components(ccs::BipartiteGraphConnectedComponents)

Return the number of connected components.
"""
function num_connected_components(ccs::BipartiteGraphConnectedComponents)
  length(ccs.lvs_of_component)
end

"""
    left_size(ccs::BipartiteGraphConnectedComponents, cc::Int)

Return the number of left vertices in connected component `cc`.
"""
function left_size(ccs::BipartiteGraphConnectedComponents, cc::Int)
  length(ccs.lvs_of_component[cc])
end

"""
    get_cc_matrix(g::BipartiteGraph, ccs::BipartiteGraphConnectedComponents, cc::Int; clear_edges=false)
        -> Tuple{SparseMatrixCSC,Vector{Int},Vector{Int}}

Extract the subgraph corresponding to connected component `cc` as a sparse
matrix.

The returned tuple contains:
- the sparse weighted adjacency matrix of the component, with local left/right
  numbering,
- the map from local row indices back to global left-vertex ids,
- the map from local column indices back to global right-vertex ids.

If `clear_edges=true`, the consumed adjacency lists are emptied from `g` as the
matrix is assembled.
"""
@timeit function get_cc_matrix(
  g::BipartiteGraph{L,R,C},
  ccs::BipartiteGraphConnectedComponents,
  cc::Int;
  clear_edges::Bool=false,
)::Tuple{SparseMatrixCSC{C,Int},Vector{Int},Vector{Int}} where {L,R,C}
  num_edges = sum(length(g.edges_from_left[lv]) for lv in ccs.lvs_of_component[cc])

  edge_left_vertex = Vector{Int}(undef, num_edges)
  edge_right_vertex = Vector{Int}(undef, num_edges)
  edge_weight = Vector{C}(undef, num_edges)
  right_map = Vector{Int}(undef, ccs.rv_size_of_component[cc])

  pos = 1
  for (i, lv_id) in enumerate(ccs.lvs_of_component[cc])
    for (rv_id, weight) in g.edges_from_left[lv_id]
      j = ccs.component_position_of_rvs[rv_id]
      right_map[j] = rv_id

      edge_left_vertex[pos] = i
      edge_right_vertex[pos] = j
      edge_weight[pos] = weight
      pos += 1
    end

    if clear_edges
      empty!(g.edges_from_left[lv_id])
      sizehint!(g.edges_from_left[lv_id], 0)
    end
  end

  return sparse(edge_left_vertex, edge_right_vertex, edge_weight),
  ccs.lvs_of_component[cc],
  right_map
end


function terms_eq_from(n::Int)::Function
  function are_equal(op1::NTuple{N, OpID}, op2::NTuple{N, OpID})::Bool where {N}
    for i in 1:N
      op1[i].n < n && op2[i].n < n && return true
      op1[i] != op2[i] && return false
    end

    return true
  end

  return are_equal
end

"""
The input graph's right vertices should be sorted.
"""
@timeit function combine_duplicate_adjacent_right_vertices!(g::BipartiteGraph, eq::Function)::Vector{Int}
  new_positions = zeros(Int, right_size(g))

  cur, next = 1, 2
  new_positions[cur] = 1
  while next <= right_size(g)
    if eq(right_vertex(g, cur), right_vertex(g, next))
      new_positions[next] = cur
    else
      cur += 1
      new_positions[next] = cur
      g.right_vertices[cur] = g.right_vertices[next]
    end

    next += 1
  end

  resize!(g.right_vertices, cur)

  ## Re-label the left edges. TODO: Think if there may be a better place to do this, my initial guess is no.
  Threads.@threads for lv_id in 1:left_size(g)
    for (i, (rv_id, weight)) in enumerate(g.edges_from_left[lv_id])
      g.edges_from_left[lv_id][i] = new_positions[rv_id], weight
    end
  end

  return new_positions
end