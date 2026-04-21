function _minimum_vertex_cover_from_matching(
  right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}},
  matched_right_of_left::Vector{Int},
  matched_left_of_right::Vector{Int},
)::Tuple{Vector{Int},Vector{Int}}
  reached_left = falses(length(right_vertex_ids_from_left))
  reached_right = falses(length(matched_left_of_right))
  queue = Int[]

  for lv_id in 1:length(right_vertex_ids_from_left)
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
    for rv_id in right_vertex_ids_from_left[lv_id]
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

  for lv_id in 1:length(right_vertex_ids_from_left)
    !reached_left[lv_id] && push!(left_ids, lv_id)
  end

  for rv_id in 1:length(matched_left_of_right)
    reached_right[rv_id] && push!(right_ids, rv_id)
  end

  return left_ids, right_ids
end

function _hopcroft_karp_maximum_matching(
  right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}}, num_right::Int
)::Tuple{Vector{Int},Vector{Int}}
  num_left = length(right_vertex_ids_from_left)
  matched_right_of_left = zeros(Int, num_left)
  matched_left_of_right = zeros(Int, num_right)

  dist = Vector{Int}(undef, num_left)
  next_edge_id = ones(Int, num_left)
  queue = Int[]
  path_left = Int[]
  path_right = Int[]

  while true
    unmatched_distance = _hopcroft_karp_layered_bfs!(
      right_vertex_ids_from_left, matched_right_of_left, matched_left_of_right, dist, queue
    )
    unmatched_distance == typemax(Int) && break

    fill!(next_edge_id, 1)
    for lv_id in 1:num_left
      matched_right_of_left[lv_id] == 0 || continue
      _hopcroft_karp_augment_from!(
        right_vertex_ids_from_left,
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
  right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}},
  matched_right_of_left::Vector{Int},
  matched_left_of_right::Vector{Int},
  dist::Vector{Int},
  queue::Vector{Int},
)::Int
  fill!(dist, typemax(Int))
  empty!(queue)

  for lv_id in 1:length(right_vertex_ids_from_left)
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

    for rv_id in right_vertex_ids_from_left[lv_id]
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
  right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}},
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

    right_vertex_ids = right_vertex_ids_from_left[lv_id]
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

# TODO: Clean up this stuff
function minimum_vertex_cover(
  g::BipartiteGraph{L,R,C}, ccs::BipartiteGraphConnectedComponents, cc::Int
)::Tuple{Vector{Int},Vector{Int}} where {L,R,C}
  left_map = ccs.lvs_of_component[cc]
  position_of_rvs_in_component = ccs.position_of_rvs_in_component
  num_right = ccs.rv_size_of_component[cc]

  right_vertex_ids_from_left = Vector{Vector{Int}}(undef, length(left_map))

  @inbounds for (i, lv_id) in enumerate(left_map)
    global_right_vertex_ids = g.right_vertex_ids_from_left[lv_id]
    local_right_vertex_ids = Vector{Int}(undef, length(global_right_vertex_ids))
    for edge_id in eachindex(global_right_vertex_ids)
      rv_id = global_right_vertex_ids[edge_id]
      local_right_vertex_ids[edge_id] = position_of_rvs_in_component[rv_id]
    end
    right_vertex_ids_from_left[i] = local_right_vertex_ids
  end

  matched_right_of_left, matched_left_of_right = _hopcroft_karp_maximum_matching(
    right_vertex_ids_from_left, num_right
  )
  local_left_ids, local_right_ids = _minimum_vertex_cover_from_matching(
    right_vertex_ids_from_left, matched_right_of_left, matched_left_of_right
  )

  @assert issorted(local_left_ids)
  @assert issorted(local_right_ids)

  return local_left_ids, local_right_ids
end
