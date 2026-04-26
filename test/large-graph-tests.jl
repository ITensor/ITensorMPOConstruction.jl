using ITensorMPOConstruction:
  minimum_vertex_cover,
  BipartiteGraph,
  OpID,
  combine_duplicate_adjacent_right_vertices!,
  compute_connected_components,
  left_size,
  get_cc_matrix,
  num_connected_components,
  right_size

using Random
using Test
using Graphs

function test_get_connected_components(nl::Int, nr::Int, max_edges_from_left::Int)
  g = BipartiteGraph{Int,Int,Float64}(
    zeros(Int, nl), zeros(Int, nr), [Int[] for _ in 1:nl], [Float64[] for _ in 1:nl]
  )
  g_ref = SimpleGraph{Int}(nl + nr)

  for lv_id in 1:nl
    for _ in 1:rand(0:max_edges_from_left)
      rv_id = rand(1:nr)
      push!(g.right_vertex_ids_from_left[lv_id], rv_id)
      push!(g.edge_weights_from_left[lv_id], 1.0)
      add_edge!(g_ref, lv_id, nl + rv_id)
    end
  end

  ccs = compute_connected_components(g)
  ccs_ref = connected_components(g_ref)

  ## Remove the right vertices from each reference connected component.
  ref_verts = Set{Vector{Int}}()
  for verts in ccs_ref
    new_verts = Vector{Int}()
    has_edge_to_right = false
    for v in verts
      v > nl && continue
      has_edge_to_right = has_edge_to_right || !isempty(neighbors(g_ref, v))

      push!(new_verts, v)
    end

    !isempty(new_verts) && has_edge_to_right && push!(ref_verts, sort!(new_verts))
  end

  @test length(ref_verts) == num_connected_components(ccs)

  for lv_ids in ccs.lvs_of_component
    verts = sort(lv_ids)
    @test verts ∈ ref_verts
    if verts ∈ ref_verts
      pop!(ref_verts, verts)
    end
  end

  @test isempty(ref_verts)
end

"""
Test the "worst case" scenario, in which the bipartite has a zipper structure
and is fully connected.
"""
function test_get_connected_components_worst_case(nl::Int)
  g = BipartiteGraph{Int,Int,Int}(
    zeros(Int, nl), zeros(Int, nl), [Int[] for _ in 1:nl], [Int[] for _ in 1:nl]
  )
  for lv_id in 1:nl
    push!(g.right_vertex_ids_from_left[lv_id], lv_id)
    push!(g.edge_weights_from_left[lv_id], 1)
    if lv_id < nl
      push!(g.right_vertex_ids_from_left[lv_id], lv_id + 1)
      push!(g.edge_weights_from_left[lv_id], 1)
    end
  end

  ccs = compute_connected_components(g)

  @test num_connected_components(ccs) == 1

  @test left_size(ccs, 1) == nl
  @test sort(ccs.lvs_of_component[1]) == [i for i in 1:nl]

  @test ccs.rv_size_of_component[1] == nl
  @test sort(ccs.position_of_rvs_in_component) == [i for i in 1:nl]
end

function test_get_cc_matrix()
  g = BipartiteGraph{Int,Int,Float64}(
    zeros(Int, 4),
    zeros(Int, 5),
    [[2, 5], [2], [1], Int[]],
    [[1.5, -2.0], [3.0], [4.0], Float64[]],
  )

  ccs = compute_connected_components(g)

  @test num_connected_components(ccs) == 2

  W, left_map, right_map = get_cc_matrix(g, ccs, 1)
  @test left_map == [1, 2]
  @test right_map == [2, 5]
  @test Matrix(W) == [1.5 -2.0; 3.0 0.0]
  @test g.right_vertex_ids_from_left[1] == [2, 5]
  @test g.edge_weights_from_left[1] == [1.5, -2.0]
  @test g.right_vertex_ids_from_left[2] == [2]
  @test g.edge_weights_from_left[2] == [3.0]

  W, left_map, right_map = get_cc_matrix(g, ccs, 2; clear_edges=true)
  @test left_map == [3]
  @test right_map == [1]
  @test Matrix(W) == [4.0;;]
  @test isempty(g.right_vertex_ids_from_left[3])
  @test isempty(g.edge_weights_from_left[3])
  @test !isempty(g.right_vertex_ids_from_left[1])
  @test !isempty(g.edge_weights_from_left[1])
  @test !isempty(g.right_vertex_ids_from_left[2])
  @test !isempty(g.edge_weights_from_left[2])
end

function test_get_cc_matrix_duplicate_edges()
  g = BipartiteGraph{Int,Int,Float64}(
    zeros(Int, 2), zeros(Int, 2), [[1, 1, 2], [2, 2]], [[1.5, -0.5, 3.0], [4.0, -1.0]]
  )

  ccs = compute_connected_components(g)

  @test num_connected_components(ccs) == 1

  W, left_map, right_map = get_cc_matrix(g, ccs, 1)
  @test left_map == [1, 2]
  @test right_map == [1, 2]
  @test Matrix(W) == [1.0 3.0; 0.0 3.0]
end

function test_combine_duplicate_adjacent_right_vertices()
  g = BipartiteGraph{Int,Int,Float64}(
    zeros(Int, 3),
    [10, 10, 20, 20, 20, 30],
    [[1, 2, 3, 6], [4, 5], [2, 5, 6]],
    [fill(1.0, 4), fill(2.0, 2), fill(3.0, 3)],
  )

  new_positions = combine_duplicate_adjacent_right_vertices!(g, ==)

  @test new_positions == [1, 1, 2, 2, 2, 3]
  @test g.right_vertices == [10, 20, 30]
  @test g.right_vertex_ids_from_left == [[1, 1, 2, 3], [2, 2], [1, 2, 3]]
  @test g.edge_weights_from_left == [fill(1.0, 4), fill(2.0, 2), fill(3.0, 3)]
end

function brute_force_minimum_vertex_cover_size(
  right_vertex_ids_from_left::AbstractVector{<:AbstractVector{Int}}, num_right::Int
)::Int
  best = typemax(Int)
  num_left = length(right_vertex_ids_from_left)

  for left_mask in 0:((1 << num_left) - 1)
    left_count = count_ones(left_mask)
    left_count >= best && continue

    for right_mask in 0:((1 << num_right) - 1)
      cover_size = left_count + count_ones(right_mask)
      cover_size >= best && continue

      covers_all_edges = true
      for lv_id in 1:num_left
        left_selected = ((left_mask >> (lv_id - 1)) & 1) == 1
        left_selected && continue

        for rv_id in right_vertex_ids_from_left[lv_id]
          if ((right_mask >> (rv_id - 1)) & 1) == 0
            covers_all_edges = false
            break
          end
        end

        covers_all_edges || break
      end

      covers_all_edges && (best = cover_size)
    end
  end

  return best
end

function brute_force_minimum_vertex_cover_size(g::BipartiteGraph)::Int
  return brute_force_minimum_vertex_cover_size(g.right_vertex_ids_from_left, right_size(g))
end

function component_adjacency(
  g::BipartiteGraph, ccs, cc::Int
)::Tuple{Vector{Vector{Int}},Vector{Int},Vector{Int}}
  left_map = ccs.lvs_of_component[cc]
  right_map = Vector{Int}(undef, ccs.rv_size_of_component[cc])
  right_vertex_ids_from_left = Vector{Vector{Int}}(undef, length(left_map))

  for (i, lv_id) in enumerate(left_map)
    global_right_vertex_ids = g.right_vertex_ids_from_left[lv_id]
    local_right_vertex_ids = Vector{Int}(undef, length(global_right_vertex_ids))
    for edge_id in eachindex(global_right_vertex_ids)
      rv_id = global_right_vertex_ids[edge_id]
      j = ccs.position_of_rvs_in_component[rv_id]
      local_right_vertex_ids[edge_id] = j
      right_map[j] = rv_id
    end
    right_vertex_ids_from_left[i] = local_right_vertex_ids
  end

  return right_vertex_ids_from_left, left_map, right_map
end

function test_minimum_vertex_cover_case(g::BipartiteGraph)
  left_vertices_before = copy(g.left_vertices)
  right_vertices_before = copy(g.right_vertices)
  right_vertex_ids_before = deepcopy(g.right_vertex_ids_from_left)
  edge_weights_before = deepcopy(g.edge_weights_from_left)
  ccs = compute_connected_components(g)

  left_ids_by_component = Int[]
  right_ids_by_component = Int[]
  for cc in 1:num_connected_components(ccs)
    component_left_ids, component_right_ids = minimum_vertex_cover(g, ccs, cc)
    component_adj, left_map, right_map = component_adjacency(g, ccs, cc)

    @test issorted(component_left_ids)
    @test issorted(component_right_ids)
    @test allunique(component_left_ids)
    @test allunique(component_right_ids)
    @test all(lv_id -> 1 <= lv_id <= length(left_map), component_left_ids)
    @test all(rv_id -> 1 <= rv_id <= length(right_map), component_right_ids)

    left_in_cover = Set(component_left_ids)
    right_in_cover = Set(component_right_ids)
    for local_lv_id in eachindex(left_map)
      for local_rv_id in component_adj[local_lv_id]
        @test (local_lv_id in left_in_cover) || (local_rv_id in right_in_cover)
      end
    end

    @test length(component_left_ids) + length(component_right_ids) ==
      brute_force_minimum_vertex_cover_size(component_adj, length(right_map))

    append!(left_ids_by_component, (left_map[lv_id] for lv_id in component_left_ids))
    append!(right_ids_by_component, (right_map[rv_id] for rv_id in component_right_ids))
  end
  sort!(left_ids_by_component)
  sort!(right_ids_by_component)

  @test g.left_vertices == left_vertices_before
  @test g.right_vertices == right_vertices_before
  @test g.right_vertex_ids_from_left == right_vertex_ids_before
  @test g.edge_weights_from_left == edge_weights_before

  left_ids = left_ids_by_component
  right_ids = right_ids_by_component
  @test issorted(left_ids)
  @test issorted(right_ids)
  @test allunique(left_ids)
  @test allunique(right_ids)
  @test all(lv_id -> 1 <= lv_id <= left_size(g), left_ids)
  @test all(rv_id -> 1 <= rv_id <= right_size(g), right_ids)

  left_in_cover = falses(left_size(g))
  right_in_cover = falses(right_size(g))
  right_degree = zeros(Int, right_size(g))

  for lv_id in left_ids
    left_in_cover[lv_id] = true
  end

  for rv_id in right_ids
    right_in_cover[rv_id] = true
  end

  for lv_id in 1:left_size(g)
    isempty(g.right_vertex_ids_from_left[lv_id]) && @test !left_in_cover[lv_id]

    for rv_id in g.right_vertex_ids_from_left[lv_id]
      right_degree[rv_id] += 1
      @test left_in_cover[lv_id] || right_in_cover[rv_id]
    end
  end

  for rv_id in 1:right_size(g)
    right_degree[rv_id] == 0 && @test !right_in_cover[rv_id]
  end

  @test length(left_ids) + length(right_ids) == brute_force_minimum_vertex_cover_size(g)
end

function test_minimum_vertex_cover_duplicate_edges_in_component()
  g = BipartiteGraph{Int,Int,Int}(
    collect(1:4),
    collect(1:5),
    [[1, 1, 2, 2], [2, 3], [4, 4, 5], Int[]],
    [[3, -1, 4, 2], [5, 6], [7, -2, 1], Int[]],
  )

  ccs = compute_connected_components(g)
  found_duplicate_component = false

  for cc in 1:num_connected_components(ccs)
    component_adj, _, right_map = component_adjacency(g, ccs, cc)
    has_duplicate_edge = any(
      local_rvs -> length(unique(local_rvs)) < length(local_rvs), component_adj
    )
    has_duplicate_edge || continue

    found_duplicate_component = true
    left_ids, right_ids = minimum_vertex_cover(g, ccs, cc)
    @test length(left_ids) + length(right_ids) ==
      brute_force_minimum_vertex_cover_size(component_adj, length(right_map))
  end

  @test found_duplicate_component
  test_minimum_vertex_cover_case(g)
end

function test_minimum_vertex_cover()
  deterministic_cases = [
    BipartiteGraph{Int,Int,Int}(Int[], Int[], Vector{Vector{Int}}(), Vector{Vector{Int}}()),
    BipartiteGraph{Int,Int,Int}(collect(1:2), collect(1:3), [[2], Int[]], [[7], Int[]]),
    BipartiteGraph{Int,Int,Int}(
      collect(1:3), collect(1:3), [[1], [2], [3]], [[1], [1], [1]]
    ),
    BipartiteGraph{Int,Int,Int}(collect(1:1), collect(1:4), [[1, 2, 3, 4]], [[1, 2, 3, 4]]),
    BipartiteGraph{Int,Int,Int}(
      collect(1:4), collect(1:4), [[1, 2], [2], [4], Int[]], [[1, 1], [1], [1], Int[]]
    ),
    BipartiteGraph{Int,Int,Int}(
      collect(1:3), collect(1:3), [Int[], [2], Int[]], [Int[], [5], Int[]]
    ),
    BipartiteGraph{Int,Int,Int}(
      collect(1:2), collect(1:2), [[1, 1, 2], [2, 2]], [[3, -1, 4], [2, 7]]
    ),
  ]

  for g in deterministic_cases
    test_minimum_vertex_cover_case(g)
  end
  test_minimum_vertex_cover_duplicate_edges_in_component()

  rng = MersenneTwister(1234)
  for _ in 1:200
    nl = rand(rng, 0:5)
    nr = rand(rng, 0:5)

    right_vertex_ids_from_left = Vector{Vector{Int}}(undef, nl)
    edge_weights_from_left = Vector{Vector{Int}}(undef, nl)

    for lv_id in 1:nl
      num_edges_from_left = nr == 0 ? 0 : rand(rng, 0:5)
      right_vertex_ids_from_left[lv_id] = [rand(rng, 1:nr) for _ in 1:num_edges_from_left]
      edge_weights_from_left[lv_id] = [rand(rng, -3:3) for _ in 1:num_edges_from_left]
    end

    g = BipartiteGraph{Int,Int,Int}(
      collect(1:nl), collect(1:nr), right_vertex_ids_from_left, edge_weights_from_left
    )
    test_minimum_vertex_cover_case(g)
  end
end

@testset "BipartiteGraph" begin
  @testset "connected components" begin
    test_get_connected_components(4, 4, 2)
    test_get_connected_components(10, 10, 4)
    test_get_connected_components(187, 294, 18)
    test_get_connected_components(8 * 10^3, 3 * 10^6, 9 * 10^2)
    test_get_connected_components_worst_case(1000)
  end

  @testset "combine duplicates" test_combine_duplicate_adjacent_right_vertices()

  @testset "get conected component matrix" begin
    test_get_cc_matrix()
    test_get_cc_matrix_duplicate_edges()
  end

  @testset "minimum vertex cover" test_minimum_vertex_cover()
end
