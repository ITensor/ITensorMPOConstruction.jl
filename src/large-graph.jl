struct BipartiteGraph{L, R, C}
  left_vertices::Vector{L}
  right_vertices::Vector{R}
  edges_from_right::Vector{Vector{Tuple{Int, C}}}
end

left_size(g::BipartiteGraph)::Int = length(g.left_vertices)
right_size(g::BipartiteGraph)::Int = length(g.right_vertices)

left_vertex(g::BipartiteGraph, lv_id::Integer) = g.left_vertices[lv_id]
right_vertex(g::BipartiteGraph, rv_id::Integer) = g.right_vertices[rv_id]

function num_edges(g::BipartiteGraph)::Int
  return sum(length(rvs) for rvs in g.edges_from_right)
end

"""
The input graph's right vertices should be sorted.
"""
@timeit function combine_duplicate_adjacent_right_vertices!(g::BipartiteGraph, eq::Function)::Nothing
  cur, next = 1, 2
  while next <= right_size(g)
    if eq(right_vertex(g, cur), right_vertex(g, next))
      append!(g.edges_from_right[cur], g.edges_from_right[next])
    else
      cur += 1
      g.right_vertices[cur] = g.right_vertices[next]
      g.edges_from_right[cur] = g.edges_from_right[next]
    end

    next += 1
  end

  resize!(g.right_vertices, cur)
  sizehint!(g.right_vertices, cur)

  resize!(g.edges_from_right, cur)
  sizehint!(g.edges_from_right, cur)

  return nothing
end

struct BipartiteGraphConnectedComponents
  rvs_of_component::Vector{Vector{Int}}
  component_position_of_lvs::Vector{Int}
  lv_size_of_component::Vector{Int}
end

## This ignores vertices that are not connected to anything
@timeit function compute_connected_components!(g::BipartiteGraph)::BipartiteGraphConnectedComponents
  component_of_lvs = zeros(Int, left_size(g))

  lvs_of_component = Vector{Set{Int}}() # TODO: Pretty sure this could be a Vector{Vector{Int}}
  for rv_id in 1:right_size(g)
    cur_component = minimum(component_of_lvs[lv_id] for (lv_id, weight) in g.edges_from_right[rv_id] if component_of_lvs[lv_id] != 0; init=typemax(Int))

    if cur_component == typemax(Int)
      # This means we have found a new component (at least for now).
      push!(lvs_of_component, Set{Int}(lv_id for (lv_id, weight) in g.edges_from_right[rv_id]))

      for (lv_id, weight) in g.edges_from_right[rv_id]
        component_of_lvs[lv_id] = length(lvs_of_component)
      end
    else
      for (lv_id, weight) in g.edges_from_right[rv_id]
        prev_component = component_of_lvs[lv_id]
        prev_component == cur_component && continue

        if prev_component == 0
          component_of_lvs[lv_id] = cur_component
          push!(lvs_of_component[cur_component], lv_id)
          continue
        end
        
        for lv_id in lvs_of_component[prev_component]
          component_of_lvs[lv_id] = cur_component
        end

        union!(lvs_of_component[cur_component], lvs_of_component[prev_component])
        empty!(lvs_of_component[prev_component])
      end
    end
  end

  lv_size_of_component = [length(lvs) for lvs in lvs_of_component]
  rvs_of_component = [Vector{Int}() for _ in lvs_of_component]
  lvs_of_component = nothing

  for rv_id in 1:right_size(g)
    isempty(g.edges_from_right[rv_id]) && continue ## If the right vertex is not connected to anything we can skip it.
    component = component_of_lvs[g.edges_from_right[rv_id][1][1]]
    push!(rvs_of_component[component], rv_id)

    ## TODO: This is a check
    if !allequal(component_of_lvs[lv_id] for (lv_id, weight) in g.edges_from_right[rv_id])
      println("\nError $rv_id: ", [component_of_lvs[lv_id] for (lv_id, weight) in g.edges_from_left[lv_id]])
      error("Oops")
    end
  end

  ## Mutate component_of_lvs which stores the component of each left vertex
  ## into the position of each left vertex within it's component.
  current_position = zeros(Int, length(rvs_of_component))
  for lv_id in 1:left_size(g)
    component = component_of_lvs[lv_id]
    component == 0 && continue

    current_position[component] += 1
    component_of_lvs[lv_id] = current_position[component]
  end

  ## Finally remove the empty components
  rvs_of_component = [rvs for rvs in rvs_of_component if !isempty(rvs)]
  lv_size_of_component = [i for i in lv_size_of_component if i != 0]

  return BipartiteGraphConnectedComponents(rvs_of_component, component_of_lvs, lv_size_of_component)
end

num_connected_components(cc::BipartiteGraphConnectedComponents) = length(cc.rvs_of_component)

function get_cc_matrix(g::BipartiteGraph{L, R, C}, ccs::BipartiteGraphConnectedComponents, cc::Int)::Tuple{SparseMatrixCSC{C, Int}, Vector{Int}, Vector{Int}} where {L, R, C}
  num_edges = sum(length(g.edges_from_right[rv]) for rvs in ccs.rvs_of_component[cc] for rv in rvs)
  
  edge_left_vertex = Vector{Int}(undef, num_edges)
  edge_right_vertex = Vector{Int}(undef, num_edges)
  edge_weight = Vector{C}(undef, num_edges)
  left_map = zeros(Int, ccs.lv_size_of_component[cc])

  pos = 1
  for (i, rv_id) in enumerate(ccs.rvs_of_component[cc])
    for (lv_id, weight) in g.edges_from_right[rv_id]
      j = ccs.component_position_of_lvs[lv_id]
      left_map[j] = lv_id

      edge_left_vertex[pos] = j
      edge_right_vertex[pos] = i
      edge_weight[pos] = weight
      pos += 1
    end
  end

  return sparse(edge_left_vertex, edge_right_vertex, edge_weight), left_map, ccs.rvs_of_component[cc]
end

function clear_edges!(g::BipartiteGraph, rv_ids::Vector{Int})::Nothing
  for rv_id in rv_ids
    empty!(g.edges_from_right[rv_id])
  end

  return nothing
end
