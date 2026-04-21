function _minimum_vertex_cover_from_matching(
  global_right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}},
  left_map::Vector{Int},
  position_of_rvs_in_component::Vector{Int},
  matched_right_of_left::Vector{Int},
  matched_left_of_right::Vector{Int},
)::Tuple{Vector{Int},Vector{Int}}
  reached_left = falses(length(left_map))
  reached_right = falses(length(matched_left_of_right))
  queue = Int[]

  for local_lv_id in 1:length(left_map)
    if matched_right_of_left[local_lv_id] == 0
      reached_left[local_lv_id] = true
      push!(queue, local_lv_id)
    end
  end

  head = 1
  while head <= length(queue)
    local_lv_id = queue[head]
    head += 1

    global_lv_id = left_map[local_lv_id]
    matched_local_rv_id = matched_right_of_left[local_lv_id]
    for global_rv_id in global_right_vertex_ids_from_left[global_lv_id]
      local_rv_id = position_of_rvs_in_component[global_rv_id]
      local_rv_id == matched_local_rv_id && continue
      reached_right[local_rv_id] && continue

      reached_right[local_rv_id] = true

      matched_local_lv_id = matched_left_of_right[local_rv_id]
      if matched_local_lv_id != 0 && !reached_left[matched_local_lv_id]
        reached_left[matched_local_lv_id] = true
        push!(queue, matched_local_lv_id)
      end
    end
  end

  left_ids = Int[]
  right_ids = Int[]

  for local_lv_id in 1:length(left_map)
    !reached_left[local_lv_id] && push!(left_ids, local_lv_id)
  end

  for local_rv_id in 1:length(matched_left_of_right)
    reached_right[local_rv_id] && push!(right_ids, local_rv_id)
  end

  return left_ids, right_ids
end

function _hopcroft_karp_maximum_matching(
  global_right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}},
  left_map::Vector{Int},
  position_of_rvs_in_component::Vector{Int},
  num_right::Int,
)::Tuple{Vector{Int},Vector{Int}}
  num_left = length(left_map)
  matched_right_of_left = zeros(Int, num_left)
  matched_left_of_right = zeros(Int, num_right)

  dist = Vector{Int}(undef, num_left)
  next_edge_id = ones(Int, num_left)
  queue = Int[]
  path_left = Int[]
  path_right = Int[]

  while true
    unmatched_distance = _hopcroft_karp_layered_bfs!(
      global_right_vertex_ids_from_left,
      left_map,
      position_of_rvs_in_component,
      matched_right_of_left,
      matched_left_of_right,
      dist,
      queue,
    )
    unmatched_distance == typemax(Int) && break

    fill!(next_edge_id, 1)
    for lv_id in 1:num_left
      matched_right_of_left[lv_id] == 0 || continue
      _hopcroft_karp_augment_from!(
        global_right_vertex_ids_from_left,
        left_map,
        position_of_rvs_in_component,
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
  global_right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}},
  left_map::Vector{Int},
  position_of_rvs_in_component::Vector{Int},
  matched_right_of_left::Vector{Int},
  matched_left_of_right::Vector{Int},
  dist::Vector{Int},
  queue::Vector{Int},
)::Int
  fill!(dist, typemax(Int))
  empty!(queue)

  for local_lv_id in 1:length(left_map)
    if matched_right_of_left[local_lv_id] == 0
      dist[local_lv_id] = 0
      push!(queue, local_lv_id)
    end
  end

  unmatched_distance = typemax(Int)
  head = 1
  while head <= length(queue)
    local_lv_id = queue[head]
    head += 1

    next_dist = dist[local_lv_id] + 1
    next_dist >= unmatched_distance && continue

    global_lv_id = left_map[local_lv_id]
    for global_rv_id in global_right_vertex_ids_from_left[global_lv_id]
      local_rv_id = position_of_rvs_in_component[global_rv_id]
      matched_local_lv_id = matched_left_of_right[local_rv_id]
      if matched_local_lv_id == 0
        unmatched_distance = next_dist
      elseif dist[matched_local_lv_id] == typemax(Int)
        dist[matched_local_lv_id] = next_dist
        push!(queue, matched_local_lv_id)
      end
    end
  end

  return unmatched_distance
end

function _hopcroft_karp_augment_from!(
  global_right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}},
  left_map::Vector{Int},
  position_of_rvs_in_component::Vector{Int},
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

    global_lv_id = left_map[lv_id]
    global_right_vertex_ids = global_right_vertex_ids_from_left[global_lv_id]
    while next_edge_id[lv_id] <= length(global_right_vertex_ids)
      global_rv_id = global_right_vertex_ids[next_edge_id[lv_id]]
      next_edge_id[lv_id] += 1

      local_rv_id = position_of_rvs_in_component[global_rv_id]
      matched_lv_id = matched_left_of_right[local_rv_id]
      if matched_lv_id == 0
        dist[lv_id] + 1 == unmatched_distance || continue

        cur_rv_id = local_rv_id
        for i in length(path_left):-1:1
          cur_lv_id = path_left[i]
          matched_right_of_left[cur_lv_id] = cur_rv_id
          matched_left_of_right[cur_rv_id] = cur_lv_id
          i > 1 && (cur_rv_id = path_right[i - 1])
        end

        return true
      end

      if dist[matched_lv_id] == dist[lv_id] + 1
        push!(path_right, local_rv_id)
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

function minimum_vertex_cover(
  g::BipartiteGraph{L,R,C}, ccs::BipartiteGraphConnectedComponents, cc::Int
)::Tuple{Vector{Int},Vector{Int}} where {L,R,C}
  left_map = ccs.lvs_of_component[cc]
  position_of_rvs_in_component = ccs.position_of_rvs_in_component
  num_right = ccs.rv_size_of_component[cc]

  matched_right_of_left, matched_left_of_right = _hopcroft_karp_maximum_matching(
    g.right_vertex_ids_from_left, left_map, position_of_rvs_in_component, num_right
  )
  local_left_ids, local_right_ids = _minimum_vertex_cover_from_matching(
    g.right_vertex_ids_from_left,
    left_map,
    position_of_rvs_in_component,
    matched_right_of_left,
    matched_left_of_right,
  )

  @assert issorted(local_left_ids)
  @assert issorted(local_right_ids)

  return local_left_ids, local_right_ids
end
