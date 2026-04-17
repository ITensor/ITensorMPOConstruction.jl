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
- `right_vertex_ids_from_left`: adjacency list from each left vertex to the
  connected right-vertex ids.
- `edge_weights_from_left`: edge weights stored in parallel with
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
  return sum(length(rvs) for rvs in g.right_vertex_ids_from_left)
end

"""
    minimum_vertex_cover(g::BipartiteGraph) -> Tuple{Vector{Int},Vector{Int}}

Return a minimum-cardinality vertex cover of the unweighted bipartite graph
underlying `g`.

The result is a pair `(left_ids, right_ids)` of ascending global vertex ids on
the left and right sides whose union touches every edge. Edge weights are
ignored.
"""
function minimum_vertex_cover(g::BipartiteGraph)::Tuple{Vector{Int},Vector{Int}}
  matched_right_of_left, matched_left_of_right = _hopcroft_karp_maximum_matching(g)

  reached_left = falses(left_size(g))
  reached_right = falses(right_size(g))
  queue = Int[]

  for lv_id in 1:left_size(g)
    if matched_right_of_left[lv_id] == 0
      reached_left[lv_id] = true
      push!(queue, lv_id)
    end
  end

  head = 1
  while head <= length(queue)
    lv_id = queue[head]
    head += 1

    matched_rv_id = matched_right_of_left[lv_id]
    for rv_id in g.right_vertex_ids_from_left[lv_id]
      rv_id == matched_rv_id && continue
      reached_right[rv_id] && continue

      reached_right[rv_id] = true

      matched_lv_id = matched_left_of_right[rv_id]
      if matched_lv_id != 0 && !reached_left[matched_lv_id]
        reached_left[matched_lv_id] = true
        push!(queue, matched_lv_id)
      end
    end
  end

  left_ids = Int[]
  right_ids = Int[]

  for lv_id in 1:left_size(g)
    !reached_left[lv_id] && push!(left_ids, lv_id)
  end

  for rv_id in 1:right_size(g)
    reached_right[rv_id] && push!(right_ids, rv_id)
  end

  return left_ids, right_ids
end

function _hopcroft_karp_maximum_matching(
  g::BipartiteGraph
)::Tuple{Vector{Int},Vector{Int}}
  matched_right_of_left = zeros(Int, left_size(g))
  matched_left_of_right = zeros(Int, right_size(g))

  dist = Vector{Int}(undef, left_size(g))
  next_edge_id = ones(Int, left_size(g))
  queue = Int[]
  path_left = Int[]
  path_right = Int[]

  while true
    unmatched_distance = _hopcroft_karp_layered_bfs!(
      g, matched_right_of_left, matched_left_of_right, dist, queue
    )
    unmatched_distance == typemax(Int) && break

    fill!(next_edge_id, 1)
    for lv_id in 1:left_size(g)
      matched_right_of_left[lv_id] == 0 || continue
      _hopcroft_karp_augment_from!(
        g,
        lv_id,
        unmatched_distance,
        matched_right_of_left,
        matched_left_of_right,
        dist,
        next_edge_id,
        path_left,
        path_right,
      )
    end
  end

  return matched_right_of_left, matched_left_of_right
end

function _hopcroft_karp_layered_bfs!(
  g::BipartiteGraph,
  matched_right_of_left::Vector{Int},
  matched_left_of_right::Vector{Int},
  dist::Vector{Int},
  queue::Vector{Int},
)::Int
  fill!(dist, typemax(Int))
  empty!(queue)

  for lv_id in 1:left_size(g)
    if matched_right_of_left[lv_id] == 0
      dist[lv_id] = 0
      push!(queue, lv_id)
    end
  end

  unmatched_distance = typemax(Int)
  head = 1
  while head <= length(queue)
    lv_id = queue[head]
    head += 1

    next_dist = dist[lv_id] + 1
    next_dist >= unmatched_distance && continue

    for rv_id in g.right_vertex_ids_from_left[lv_id]
      matched_lv_id = matched_left_of_right[rv_id]
      if matched_lv_id == 0
        unmatched_distance = next_dist
      elseif dist[matched_lv_id] == typemax(Int)
        dist[matched_lv_id] = next_dist
        push!(queue, matched_lv_id)
      end
    end
  end

  return unmatched_distance
end

function _hopcroft_karp_augment_from!(
  g::BipartiteGraph,
  start_lv_id::Int,
  unmatched_distance::Int,
  matched_right_of_left::Vector{Int},
  matched_left_of_right::Vector{Int},
  dist::Vector{Int},
  next_edge_id::Vector{Int},
  path_left::Vector{Int},
  path_right::Vector{Int},
)::Bool
  empty!(path_left)
  empty!(path_right)
  push!(path_left, start_lv_id)

  ## Search the layered graph iteratively to avoid deep recursion on large inputs.
  while !isempty(path_left)
    lv_id = path_left[end]
    found_step = false

    right_vertex_ids = g.right_vertex_ids_from_left[lv_id]
    while next_edge_id[lv_id] <= length(right_vertex_ids)
      rv_id = right_vertex_ids[next_edge_id[lv_id]]
      next_edge_id[lv_id] += 1

      matched_lv_id = matched_left_of_right[rv_id]
      if matched_lv_id == 0
        dist[lv_id] + 1 == unmatched_distance || continue

        cur_rv_id = rv_id
        for i in length(path_left):-1:1
          cur_lv_id = path_left[i]
          matched_right_of_left[cur_lv_id] = cur_rv_id
          matched_left_of_right[cur_rv_id] = cur_lv_id
          i > 1 && (cur_rv_id = path_right[i - 1])
        end

        return true
      end

      if dist[matched_lv_id] == dist[lv_id] + 1
        push!(path_right, rv_id)
        push!(path_left, matched_lv_id)
        found_step = true
        break
      end
    end

    found_step && continue

    dist[lv_id] = typemax(Int)
    pop!(path_left)
    !isempty(path_right) && pop!(path_right)
  end

  return false
end

"""
    clear_edges_from_left!(g::BipartiteGraph, lv_id::Integer) -> Nothing

Remove all edges incident on the left vertex `lv_id`.

This clears both the adjacent right-vertex ids and their corresponding edge
weights, and releases any retained capacity in those per-vertex adjacency
lists.
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

The right vertices are assumed to already be grouped so that duplicate vertices
appear contiguously. The first vertex of each equal run is kept, later vertices
in the run are removed, and every right-vertex id stored in the left adjacency
lists is remapped to the surviving vertex id. The returned vector maps each
original right-vertex id to its new position after compaction.
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
function compute_connected_components(g::BipartiteGraph)::BipartiteGraphConnectedComponents
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

    for rv_id in g.right_vertex_ids_from_left[lv_id]
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
    end
  end

  ## Mutate min_lv_connected_to_rv which stores the first left vertex connected to each
  ## right vertex into the position of each right vertex within it's component.
  rv_size_of_component = zeros(Int, length(lvs_of_component))
  @inbounds for rv_id in 1:right_size(g)
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
function get_cc_matrix(
  g::BipartiteGraph{L,R,C},
  ccs::BipartiteGraphConnectedComponents,
  cc::Int;
  clear_edges::Bool=false,
)::Tuple{SparseMatrixCSC{C,Int},Vector{Int},Vector{Int}} where {L,R,C}
  left_map = ccs.lvs_of_component[cc]
  num_left = length(left_map)
  num_right = ccs.rv_size_of_component[cc]
  component_position_of_rvs = ccs.component_position_of_rvs

  ## Count entries in each column and record the local-to-global right-vertex map.
  colptr = zeros(Int, num_right + 1)
  right_map = Vector{Int}(undef, num_right)
  @inbounds for lv_id in left_map
    for rv_id in g.right_vertex_ids_from_left[lv_id]
      j = component_position_of_rvs[rv_id]
      colptr[j + 1] += 1
      right_map[j] = rv_id
    end
  end

  colptr[1] = 1
  @inbounds for j in 1:num_right
    colptr[j + 1] += colptr[j]
  end

  num_edges = colptr[end] - 1
  rowvals = Vector{Int}(undef, num_edges)
  nzvals = Vector{C}(undef, num_edges)
  next_position = copy(colptr)

  ## Fill the CSC backing storage directly. Since the rows are traversed in
  ## increasing order, each column is already row-sorted.
  @inbounds for (i, lv_id) in enumerate(left_map)
    right_vertex_ids = g.right_vertex_ids_from_left[lv_id]
    edge_weights = g.edge_weights_from_left[lv_id]
    for edge_id in eachindex(right_vertex_ids)
      rv_id = right_vertex_ids[edge_id]
      weight = edge_weights[edge_id]
      j = component_position_of_rvs[rv_id]
      pos = next_position[j]
      rowvals[pos] = i
      nzvals[pos] = weight
      next_position[j] = pos + 1
    end

    if clear_edges
      clear_edges_from_left!(g, lv_id)
    end
  end

  ## Merge duplicate structural entries so repeated edges keep the same
  ## semantics as `sparse(row, col, val)`.
  write_pos = 1
  @inbounds for j in 1:num_right
    start = colptr[j]
    stop = colptr[j + 1] - 1
    colptr[j] = write_pos
    start > stop && continue

    row = rowvals[start]
    val = nzvals[start]
    for pos in (start + 1):stop
      if rowvals[pos] == row
        val += nzvals[pos]
      else
        rowvals[write_pos] = row
        nzvals[write_pos] = val
        write_pos += 1
        row = rowvals[pos]
        val = nzvals[pos]
      end
    end

    rowvals[write_pos] = row
    nzvals[write_pos] = val
    write_pos += 1
  end
  colptr[end] = write_pos
  resize!(rowvals, write_pos - 1)
  resize!(nzvals, write_pos - 1)

  return SparseMatrixCSC(num_left, num_right, colptr, rowvals, nzvals), left_map, right_map
end
