using ITensorMPOConstruction:
  BipartiteGraph,
  OpID,
  combine_duplicate_adjacent_right_vertices!,
  compute_connected_components,
  left_size,
  get_cc_matrix,
  num_connected_components

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
  @test sort(ccs.component_position_of_rvs) == [i for i in 1:nl]
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
    zeros(Int, 2),
    zeros(Int, 2),
    [[1, 1, 2], [2, 2]],
    [[1.5, -0.5, 3.0], [4.0, -1.0]],
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

@testset "BipartiteGraph" begin
  test_get_connected_components(4, 4, 2)
  test_get_connected_components(10, 10, 4)
  test_get_connected_components(187, 294, 18)
  test_get_connected_components(8 * 10^3, 3 * 10^6, 9 * 10^2)
  test_get_connected_components_worst_case(1000)
  test_combine_duplicate_adjacent_right_vertices()
  test_get_cc_matrix()
  test_get_cc_matrix_duplicate_edges()
end
