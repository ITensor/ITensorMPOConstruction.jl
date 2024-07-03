struct BipartiteGraph{L, R, C}
  left_vertices::Vector{L}
  right_vertices::Vector{R}
  edges_from_left::Vector{Vector{Tuple{Int, C}}}
end

left_size(g::BipartiteGraph)::Int = length(g.left_vertices)
right_size(g::BipartiteGraph)::Int = length(g.right_vertices)

left_vertex(g::BipartiteGraph, lv_id::Integer) = g.left_vertices[lv_id]
right_vertex(g::BipartiteGraph, rv_id::Integer) = g.right_vertices[rv_id]

"""
The input graph's right vertices should be sorted.
"""
function combine_duplicate_adjacent_right_vertices!(g::BipartiteGraph, eq::Function)::Nothing
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

function compute_connected_components!(g::BipartiteGraph)::Nothing
  component_of_rv = zeros(Int, right_size(g)) # TODO: IDK if I should allocate this each time, also could probably be Int16

  rvs_of_component = Vector{Set{Int}}() # TODO: Pretty sure this could be a Vector{Vector{Int}}
  for lv_id in 1:left_size(g)
    cur_component = minimum(component_of_rv[rv_id] for (rv_id, weight) in g.edges_from_left[lv_id] if component_of_rv[rv_id] != 0; init=typemax(Int))

    if cur_component == typemax(Int)
      # This means we have found a new component (at least for now).
      push!(rvs_of_component, Set{Int}(rv_id for (rv_id, weight) in g.edges_from_left[lv_id]))

      for (rv_id, weight) in g.edges_from_left[lv_id]
        component_of_rv[rv_id] = length(rvs_of_component)
      end
    else
      for (rv_id, weight) in g.edges_from_left[lv_id]
        prev_component = component_of_rv[rv_id]
        prev_component == cur_component && continue

        if prev_component == 0
          component_of_rv[rv_id] = cur_component
          push!(rvs_of_component[cur_component], rv_id)
          continue
        end
        
        for rv_id in rvs_of_component[prev_component]
          component_of_rv[rv_id] = cur_component
        end

        union!(rvs_of_component[cur_component], rvs_of_component[prev_component])
        empty!(rvs_of_component[prev_component])
      end
    end
  end

  ## TODO: This is a check
  for lv_id in 1:left_size(g)
    if !allequal(component_of_rv[rv_id] for (rv_id, weight) in g.edges_from_left[lv_id])
      println("\nError $lv_id: ", [component_of_rv[rv_id] for (rv_id, weight) in g.edges_from_left[lv_id]])
      error("Oops")
    end
  end

  return nothing
end

num_connected_components(g::BipartiteGraph) = 1

function get_cc_matrix(g::BipartiteGraph{L, R, C}, cc::Int)::Tuple{SparseMatrixCSC{C, Int}, Vector{Int}, Vector{Int}} where {L, R, C}
  @assert cc == 1

  edge_left_vertex = Int[]
  edge_right_vertex = Int[]
  edge_weight = C[]

  for lv_id in 1:left_size(g)
    for (rv_id, weight) in g.edges_from_left[lv_id]
      push!(edge_left_vertex, lv_id)
      push!(edge_right_vertex, rv_id)
      push!(edge_weight, weight)
    end
  end

  return sparse(edge_right_vertex, edge_left_vertex, edge_weight), [i for i in 1:left_size(g)], [i for i in 1:right_size(g)]
end

function clear_edges!(g::BipartiteGraph, left_ids::Vector{Int})::Nothing
  for left_id in left_ids
    empty!(g.edges_from_left[left_id])
  end

  return nothing
end
