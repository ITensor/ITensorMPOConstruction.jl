"""
    BipartiteGraph{L,R,C}

A weighted bipartite graph with typed left and right vertex payloads.

Left and right vertex ids are the positions of those payloads in
`left_vertices` and `right_vertices`. Edges are stored as per-left-vertex
adjacency lists: `right_vertex_ids_from_left[lv_id][edge_id]` gives the
right-vertex id for one stored edge entry, and
`edge_weights_from_left[lv_id][edge_id]` gives its corresponding weight.
Duplicate right-vertex ids may appear in a single adjacency list.

Type Parameters
- `L`: The type of the left vertices.
- `R`: The type of the right vertices.
- `C`: The scalar edge weight type.

Fields:
- `left_vertices`: metadata stored for each left vertex.
- `right_vertices`: metadata stored for each right vertex.
- `right_vertex_ids_from_left`: adjacency list from each left vertex to the
  connected right-vertex ids.
- `edge_weights_from_left`: edge weights stored entry-by-entry in parallel with
  `right_vertex_ids_from_left`.
"""
struct BipartiteGraph{L,R,C}
  left_vertices::Vector{L}
  right_vertices::Vector{R}
  right_vertex_ids_from_left::Vector{Vector{Int}}
  edge_weights_from_left::Vector{Vector{C}}
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

Return the payload associated with the left vertex id `lv_id`.
"""
left_vertex(g::BipartiteGraph, lv_id::Integer) = g.left_vertices[lv_id]

"""
    right_vertex(g::BipartiteGraph, rv_id::Integer)

Return the payload associated with the right vertex id `rv_id`.
"""
right_vertex(g::BipartiteGraph, rv_id::Integer) = g.right_vertices[rv_id]

"""
    num_edges(g::BipartiteGraph) -> Int

Return the number of stored edge entries in `g`.
"""
function num_edges(g::BipartiteGraph)::Int
  return sum(length(rvs) for rvs in g.right_vertex_ids_from_left)
end

"""
    weighted_edge_iterator(g::BipartiteGraph, lv_id::Integer)

Return a lazy iterator over the edges from the left vertex id `lv_id`.

Each item is a `(rv_id, weight)` pair, where `rv_id` is a right vertex
id and `weight` is the corresponding edge weight. The iterator views the
graph's parallel adjacency and weight vectors directly rather than copying
them.
"""
function weighted_edge_iterator(g::BipartiteGraph, lv_id::Integer)
  return zip(g.right_vertex_ids_from_left[lv_id], g.edge_weights_from_left[lv_id])
end

"""
    clear_edges_from_left!(g::BipartiteGraph, lv_id::Integer) -> Nothing

Remove all stored edge entries incident on the left vertex id `lv_id`.

This clears both the adjacent right-vertex ids and their corresponding edge
weights, and releases retained capacity in those per-vertex adjacency lists.
"""
@inline function clear_edges_from_left!(g::BipartiteGraph, lv_id::Integer)::Nothing
  empty!(g.right_vertex_ids_from_left[lv_id])
  sizehint!(g.right_vertex_ids_from_left[lv_id], 0)
  empty!(g.edge_weights_from_left[lv_id])
  sizehint!(g.edge_weights_from_left[lv_id], 0)
  return nothing
end

"""
    combine_duplicate_adjacent_right_vertices!(g::BipartiteGraph, eq::Function) -> Vector{Int}

Combine adjacent right vertices in `g` that compare equal under `eq`.

The right vertices are assumed to already be grouped so that duplicate payloads
appear contiguously. The
predicate `eq` is called on adjacent right-vertex payloads. The first vertex of
each equal run is kept, later vertices in the run are removed, and every
right-vertex id stored in the left adjacency lists is remapped to the surviving
right-vertex id. Edge weights are left unchanged, so duplicate edge
entries may remain after the remapping. The returned vector maps each original
right-vertex id to its new position after compaction.
"""
@timeit function combine_duplicate_adjacent_right_vertices!(
  g::BipartiteGraph, eq::Function
)::Vector{Int}
  right_vertices = g.right_vertices

  starts_new_run = Vector{Int}(undef, length(right_vertices))
  starts_new_run[1] = true
  Threads.@threads for rv_id in 2:length(right_vertices)
    @inbounds starts_new_run[rv_id] = !eq(right_vertices[rv_id - 1], right_vertices[rv_id])
  end

  let cur = 0
    @inbounds for rv_id in eachindex(right_vertices)
      if starts_new_run[rv_id] == true
        cur += 1
        right_vertices[cur] = right_vertices[rv_id]
      end
      starts_new_run[rv_id] = cur
    end
  end

  # starts_new_run now contains the position of the first equal right vertex.
  new_positions = starts_new_run

  resize!(right_vertices, new_positions[end])

  Threads.@threads for lv_id in 1:left_size(g)
    right_vertex_ids = g.right_vertex_ids_from_left[lv_id]
    for i in eachindex(right_vertex_ids)
      right_vertex_ids[i] = new_positions[right_vertex_ids[i]]
    end
  end

  return new_positions
end
