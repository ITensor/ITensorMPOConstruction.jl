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

struct BipartiteGraphConnectedComponents
  lvs_of_component::Vector{Vector{Int}}
  component_position_of_rvs::Vector{Int}
  rv_size_of_component::Vector{Int}
end

@timeit function compute_connected_components(g::BipartiteGraph)::BipartiteGraphConnectedComponents
  min_lv_connected_to_rv = fill(typemax(Int), right_size(g))
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

  return BipartiteGraphConnectedComponents(lvs_of_component_non_empty, min_lv_connected_to_rv, rv_size_of_component_non_empty)
end

num_connected_components(ccs::BipartiteGraphConnectedComponents) = length(ccs.lvs_of_component)

left_size(ccs::BipartiteGraphConnectedComponents, cc::Int) = length(ccs.lvs_of_component[cc])

function get_cc_matrix(g::BipartiteGraph{L, R, C}, ccs::BipartiteGraphConnectedComponents, cc::Int; clear_edges::Bool=false)::Tuple{SparseMatrixCSC{C, Int}, Vector{Int}, Vector{Int}} where {L, R, C}
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

  return sparse(edge_left_vertex, edge_right_vertex, edge_weight), ccs.lvs_of_component[cc], right_map
end

@timeit function get_default_tol(g::BipartiteGraph{L, R, C}, ccs::BipartiteGraphConnectedComponents)::Float64 where {L, R, C}
  column_norms = zeros(right_size(g))
  for edges in g.edges_from_left
    for (rv_id, weight) in edges
      column_norms[rv_id] += abs2(weight)
    end
  end

  # Taken from https://fossies.org/linux/SuiteSparse/SPQR/Doc/spqr_user_guide.pdf Section 2.3: The opts parameter
  # It is possible that there are right vertices that are not connected to anything on the left. This corresponds
  # to the truncation dropping terms wholesale.
  num_rows = left_size(g)
  num_columns = sum(ccs.rv_size_of_component)

  return 20 * (num_rows + num_columns) * eps() * sqrt(maximum(column_norms))
end
