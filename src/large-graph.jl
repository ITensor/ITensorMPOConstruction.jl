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

## TODO: document
@timeit function combine_duplicate_adjacent_right_vertices!(g::BipartiteGraph, eq::Function)::Vector{Int}
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
    for (i, (rv_id, weight)) in enumerate(g.edges_from_left[lv_id])
      g.edges_from_left[lv_id][i] = new_positions[rv_id], weight
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
function compute_connected_components(
  g::BipartiteGraph
)::BipartiteGraphConnectedComponents
  return compute_connected_components(g, Vector{Int}(undef, right_size(g)))
end

# @inline function _find_component_root!(parent::Vector{Int}, lv_id::Int)::Int
#   root = lv_id
#   @inbounds while parent[root] != root
#     root = parent[root]
#   end

#   @inbounds while parent[lv_id] != root
#     next_lv = parent[lv_id]
#     parent[lv_id] = root
#     lv_id = next_lv
#   end

#   return root
# end

# @inline function _merge_component_anchors!(
#   parent::Vector{Int},
#   next_lv_in_component::Vector{Int},
#   tail_of_component::Vector{Int},
#   left_size_of_component::Vector{Int},
#   current_anchor_lv::Int,
#   other_anchor_lv::Int,
# )::Int
#   current_anchor_lv, other_anchor_lv = minmax(current_anchor_lv, other_anchor_lv)
#   current_root = _find_component_root!(parent, current_anchor_lv)
#   other_root = _find_component_root!(parent, other_anchor_lv)
#   current_root == other_root && return current_anchor_lv

#   @inbounds begin
#     parent[other_root] = current_root
#     next_lv_in_component[tail_of_component[current_root]] = other_root
#     tail_of_component[current_root] = tail_of_component[other_root]
#     left_size_of_component[current_root] += left_size_of_component[other_root]
#   end
#   return current_anchor_lv
# end

# @timeit function compute_connected_components(
#   g::BipartiteGraph, workspace::Vector{Int}
# )::BipartiteGraphConnectedComponents
#   nl = left_size(g)
#   nr = right_size(g)
#   unseen_rv = typemax(Int)

#   @assert length(workspace) >= nr
#   resize!(workspace, nr)
#   first_lv_connected_to_rv = workspace
#   fill!(first_lv_connected_to_rv, unseen_rv)

#   parent = Vector{Int}(undef, nl)
#   next_lv_in_component = zeros(Int, nl)
#   tail_of_component = Vector{Int}(undef, nl)
#   left_size_of_component = ones(Int, nl)
#   @timeit "1" @inbounds for lv_id in 1:nl
#     parent[lv_id] = lv_id
#     tail_of_component[lv_id] = lv_id
#   end

#   @timeit "2" @inbounds for lv_id in 1:nl
#     current_anchor_lv = lv_id
#     for (rv_id, _) in g.edges_from_left[lv_id]
#       first_lv = first_lv_connected_to_rv[rv_id]
#       if first_lv == unseen_rv
#         first_lv_connected_to_rv[rv_id] = lv_id
#         continue
#       end

#       current_anchor_lv = _merge_component_anchors!(
#         parent,
#         next_lv_in_component,
#         tail_of_component,
#         left_size_of_component,
#         current_anchor_lv,
#         first_lv,
#       )
#     end
#   end

#   num_right_of_component = zeros(Int, nl)
#   @timeit "3" @inbounds for rv_id in 1:nr
#     first_lv = first_lv_connected_to_rv[rv_id]
#     first_lv == unseen_rv && continue

#     root = _find_component_root!(parent, first_lv)
#     first_lv_connected_to_rv[rv_id] = root
#     num_right_of_component[root] += 1
#   end

#   rv_size_of_component = Int[]
#   lvs_of_component = Vector{Vector{Int}}()
#   @timeit "4" @inbounds for root in 1:nl
#     parent[root] != root && continue
#     num_right = num_right_of_component[root]
#     num_right == 0 && continue

#     push!(rv_size_of_component, num_right)
#     lvs = Vector{Int}(undef, left_size_of_component[root])
#     push!(lvs_of_component, lvs)

#     lv_id = root
#     pos = 1
#     while lv_id != 0
#       lvs[pos] = lv_id
#       lv_id = next_lv_in_component[lv_id]
#       pos += 1
#     end

#     num_right_of_component[root] = 0
#   end

#   @timeit "5" @inbounds for rv_id in 1:nr
#     root = first_lv_connected_to_rv[rv_id]
#     root == unseen_rv && continue

#     num_right_of_component[root] += 1
#     first_lv_connected_to_rv[rv_id] = num_right_of_component[root]
#   end

#   return BipartiteGraphConnectedComponents(
#     lvs_of_component, first_lv_connected_to_rv, rv_size_of_component
#   )
# end

@timeit function compute_connected_components(
  g::BipartiteGraph, workspace::Vector{Int}
)::BipartiteGraphConnectedComponents
  @assert length(workspace) >= right_size(g)
  resize!(workspace, right_size(g))
  min_lv_connected_to_rv = workspace
  min_lv_connected_to_rv .= typemax(Int)

  component_of_lv = Int[i for i in 1:left_size(g)]
  @timeit "min_lv_connected_to_rv" let
    changed = true
    while changed
      changed = false
      Threads.@threads for lv_id in 1:left_size(g)
        @inbounds for (rv_id, _) in g.edges_from_left[lv_id]
          ret = cmp(component_of_lv[lv_id], min_lv_connected_to_rv[rv_id])
          ret == 0 && continue
          
          changed = true
          if ret < 0
            min_lv_connected_to_rv[rv_id] = component_of_lv[lv_id]
          else
            component_of_lv[lv_id] = min_lv_connected_to_rv[rv_id]
          end
        end
      end
    end
  end

  lvs_of_component = [Vector{Int}() for _ in 1:maximum(component_of_lv)]
  @timeit "lvs_of_component" for (lv_id, c) in enumerate(component_of_lv)
    push!(lvs_of_component[c], lv_id)
  end

  ## Mutate min_lv_connected_to_rv which stores the first left vertex connected to each
  ## right vertex into the position of each right vertex within it's component.
  rv_size_of_component = zeros(Int, length(lvs_of_component))
  @timeit "2" @inbounds for rv_id in 1:right_size(g)
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
  @timeit "3" for (i, lvs) in enumerate(lvs_of_component)
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
    for (rv_id, _) in g.edges_from_left[lv_id]
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
    for (rv_id, weight) in g.edges_from_left[lv_id]
      j = component_position_of_rvs[rv_id]
      pos = next_position[j]
      rowvals[pos] = i
      nzvals[pos] = weight
      next_position[j] = pos + 1
    end

    if clear_edges
      empty!(g.edges_from_left[lv_id])
      sizehint!(g.edges_from_left[lv_id], 0)
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
