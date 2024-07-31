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
  sizehint!(g.right_vertices, cur)

  for lv_id in 1:left_size(g)
    for (i, (rv_id, weight)) in enumerate(g.edges_from_left[lv_id])
      g.edges_from_left[lv_id][i] = new_positions[rv_id], weight
    end
  end

  return nothing
end

struct BipartiteGraphConnectedComponents
  lvs_of_component::Vector{Vector{Int}}
end

function compute_connected_components2!(g::BipartiteGraph)::BipartiteGraphConnectedComponents
  min_lv_connected_to_rv = fill(typemax(Int32), right_size(g))
  component_of_lv = Int32[i for i in 1:left_size(g)]
  lvs_of_component = Vector{Int}[[i] for i in 1:left_size(g)]
  
  for lv_id in 1:left_size(g)
    cur_min_lv = lv_id

    for (rv_id, _) in g.edges_from_left[lv_id]
      min_lv_of_rv = min_lv_connected_to_rv[rv_id]

      if min_lv_of_rv == typemax(Int32)
        min_lv_connected_to_rv[rv_id] = lv_id
        continue
      end

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

  lvs_of_component = [lvs for lvs in lvs_of_component if !isempty(lvs)]

  return BipartiteGraphConnectedComponents(lvs_of_component)
end

function compute_connected_components3!(g::BipartiteGraph)::BipartiteGraphConnectedComponents
  ## TODO: This is a fix for DataStructures MutableLinkedList.append! that has yet to be
  ## released, should be in the next release (currently julia v0.18.20, github V0.18.15??)
  function my_append!(l1::MutableLinkedList{T}, l2::MutableLinkedList{T}) where T
    l1.node.prev.next = l2.node.next # l1's last's next is now l2's first
    l2.node.prev.next = l1.node # l2's last's next is now l1.node
    l2.node.next.prev = l1.node.prev # l2's first's prev is now l1's last
    l1.node.prev      = l2.node.prev # l1's first's prev is now l2's last
    l1.len += length(l2)
    # make l2 empty
    l2.node.prev = l2.node
    l2.node.next = l2.node
    l2.len = 0
    return l1
  end

  min_lv_connected_to_rv = fill(typemax(Int32), right_size(g))
  component_of_lv = Int32[i for i in 1:left_size(g)]
  lvs_of_component = MutableLinkedList{Int}[MutableLinkedList{Int}(i) for i in 1:left_size(g)]
  
  for lv_id in 1:left_size(g)
    cur_min_lv = lv_id

    for (rv_id, _) in g.edges_from_left[lv_id]
      min_lv_of_rv = min_lv_connected_to_rv[rv_id]

      if min_lv_of_rv == typemax(Int32)
        min_lv_connected_to_rv[rv_id] = lv_id
        continue
      end

      cur_min_lv, src_lv = minmax(cur_min_lv, min_lv_of_rv)
      cur_component = component_of_lv[cur_min_lv]
      src_component = component_of_lv[src_lv]

      cur_component == src_component && continue

      for lv_id in lvs_of_component[src_component]
        component_of_lv[lv_id] = cur_component
      end

      my_append!(lvs_of_component[cur_component], lvs_of_component[src_component])
    end
  end

  lvs_of_component = [collect(lvs) for lvs in lvs_of_component if !isempty(lvs)]

  return BipartiteGraphConnectedComponents(lvs_of_component)
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

  ## Finally remove the empty components
  lvs_of_component = [lvs for lvs in lvs_of_component if !isempty(lvs)]

  return BipartiteGraphConnectedComponents(lvs_of_component)
end

num_connected_components(cc::BipartiteGraphConnectedComponents) = length(cc.lvs_of_component)

function get_cc_matrix(g::BipartiteGraph{L, R, C}, ccs::BipartiteGraphConnectedComponents, cc::Int; clear_edges::Bool=false)::Tuple{SparseMatrixCSC{C, Int}, Vector{Int}, Vector{Int}} where {L, R, C}
  num_edges = sum(length(g.edges_from_left[lv]) for lvs in ccs.lvs_of_component[cc] for lv in lvs)

  edge_left_vertex = Vector{Int}(undef, num_edges)
  edge_right_vertex = Vector{Int}(undef, num_edges)
  edge_weight = Vector{C}(undef, num_edges)

  right_map_inv = Dict{Int, Int}()
  right_map = Vector{Int}()

  pos = 1
  for (i, lv_id) in enumerate(ccs.lvs_of_component[cc])
    for (rv_id, weight) in g.edges_from_left[lv_id]
      j = get!(right_map_inv, rv_id) do 
        push!(right_map, rv_id)
        return length(right_map)
      end

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
