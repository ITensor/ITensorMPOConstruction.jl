struct BipartiteGraph{L, R, C}
  left_vertices::Vector{L}
  right_vertices::Vector{R}
  edges_from_left::Vector{Vector{Tuple{Int, C}}}
end

left_size(g::BipartiteGraph)::Int = length(g.left_vertices)
right_size(g::BipartiteGraph)::Int = length(g.right_vertices)

left_vertex(g::BipartiteGraph, lv_id::Integer) = g.left_vertices[lv_id]
right_vertex(g::BipartiteGraph, rv_id::Integer) = g.right_vertices[rv_id]

function num_edges(g::BipartiteGraph)::Int
  return sum(length(rvs) for rvs in g.edges_from_left)
end

"""
The input graph's right vertices should be sorted.
"""
@timeit function combine_duplicate_adjacent_right_vertices!(g::BipartiteGraph, eq::Function)::Nothing
  new_positions = zeros(Int, right_size(g)) # TODO: IDK if I should allocate this each time

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
  for lv_id in 1:left_size(g)
    for (i, (rv_id, weight)) in enumerate(g.edges_from_left[lv_id])
      g.edges_from_left[lv_id][i] = new_positions[rv_id], weight
    end
  end

  return nothing
end

struct BipartiteGraphConnectedComponents
  lvs_of_component::Vector{Vector{Int}}
  component_position_of_rvs::Vector{Int}
  rv_size_of_component::Vector{Int}
end

## This ignores vertices that are not connected to anything
@timeit function compute_connected_components!(g::BipartiteGraph)::BipartiteGraphConnectedComponents
  component_of_rvs = zeros(Int, right_size(g)) # TODO: IDK if I should allocate this each time, also could probably be Int16

  rvs_of_component = Vector{Set{Int}}() # TODO: Pretty sure this could be a Vector{Vector{Int}}
  for lv_id in 1:left_size(g)
    cur_component = minimum(component_of_rvs[rv_id] for (rv_id, weight) in g.edges_from_left[lv_id] if component_of_rvs[rv_id] != 0; init=typemax(Int))

    if cur_component == typemax(Int)
      # This means we have found a new component (at least for now).
      push!(rvs_of_component, Set{Int}(rv_id for (rv_id, weight) in g.edges_from_left[lv_id]))

      for (rv_id, weight) in g.edges_from_left[lv_id]
        component_of_rvs[rv_id] = length(rvs_of_component)
      end
    else
      for (rv_id, weight) in g.edges_from_left[lv_id]
        prev_component = component_of_rvs[rv_id]
        prev_component == cur_component && continue

        if prev_component == 0
          component_of_rvs[rv_id] = cur_component
          push!(rvs_of_component[cur_component], rv_id)
          continue
        end
        
        for rv_id in rvs_of_component[prev_component]
          component_of_rvs[rv_id] = cur_component
        end

        union!(rvs_of_component[cur_component], rvs_of_component[prev_component])
        empty!(rvs_of_component[prev_component])
      end
    end
  end

  rv_size_of_component = [length(rvs) for rvs in rvs_of_component]
  lvs_of_component = [Vector{Int}() for _ in rvs_of_component]
  rvs_of_component = nothing

  for lv_id in 1:left_size(g)
    isempty(g.edges_from_left[lv_id]) && continue ## If the left vertex is not connected to anything we can skip it.
    component = component_of_rvs[g.edges_from_left[lv_id][1][1]]
    push!(lvs_of_component[component], lv_id)

    ## TODO: This is a check
    if !allequal(component_of_rvs[rv_id] for (rv_id, weight) in g.edges_from_left[lv_id])
      println("\nError $lv_id: ", [component_of_rvs[rv_id] for (rv_id, weight) in g.edges_from_left[lv_id]])
      error("Oops")
    end
  end

  ## Mutate component_of_rvs which stores the component of each right right vertex
  ## into the position of each right vertex within it's component.
  current_position = zeros(Int, length(lvs_of_component))
  for rv_id in 1:right_size(g)
    component = component_of_rvs[rv_id]
    component == 0 && continue

    current_position[component] += 1
    component_of_rvs[rv_id] = current_position[component]
  end

  ## Finally remove the empty components
  lvs_of_component = [lvs for lvs in lvs_of_component if !isempty(lvs)]
  rv_size_of_component = [i for i in rv_size_of_component if i != 0]

  return BipartiteGraphConnectedComponents(lvs_of_component, component_of_rvs, rv_size_of_component)
end

num_connected_components(cc::BipartiteGraphConnectedComponents) = length(cc.lvs_of_component)

function get_cc_matrix(g::BipartiteGraph{L, R, C}, ccs::BipartiteGraphConnectedComponents, cc::Int)::Tuple{SparseMatrixCSC{C, Int}, Vector{Int}, Vector{Int}} where {L, R, C}
  num_edges = sum(length(g.edges_from_left[lv]) for lvs in ccs.lvs_of_component[cc] for lv in lvs)
  
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
  end

  return sparse(edge_left_vertex, edge_right_vertex, edge_weight), ccs.lvs_of_component[cc], right_map
end

function clear_edges!(g::BipartiteGraph, left_ids::Vector{Int})::Nothing
  for left_id in left_ids
    empty!(g.edges_from_left[left_id])
  end

  return nothing
end
