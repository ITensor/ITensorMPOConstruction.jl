using ITensorMPOConstruction:
  BipartiteGraph, compute_connected_components, get_cc_matrix, num_connected_components

using Test
using Graphs

function test_get_connected_components(nl::Int, nr::Int, max_edges_from_left::Int)
  g = BipartiteGraph{Int,Int,Float64}(
    zeros(Int, nl), zeros(Int, nr), [Vector{Tuple{Int,Float64}}() for _ in 1:nl]
  )
  g_ref = SimpleGraph{Int}(nl + nr)

  for lv_id in 1:nl
    for _ in 1:rand(0:max_edges_from_left)
      rv_id = rand(1:nr)
      push!(g.edges_from_left[lv_id], (rv_id, 1.0))
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

function test_get_cc_matrix()
  g = BipartiteGraph{Int,Int,Float64}(
    zeros(Int, 4),
    zeros(Int, 5),
    [[(2, 1.5), (5, -2.0)], [(2, 3.0)], [(1, 4.0)], Tuple{Int,Float64}[]],
  )

  ccs = compute_connected_components(g)

  @test num_connected_components(ccs) == 2

  W, left_map, right_map = get_cc_matrix(g, ccs, 1)
  @test left_map == [1, 2]
  @test right_map == [2, 5]
  @test Matrix(W) == [1.5 -2.0; 3.0 0.0]
  @test g.edges_from_left[1] == [(2, 1.5), (5, -2.0)]
  @test g.edges_from_left[2] == [(2, 3.0)]

  W, left_map, right_map = get_cc_matrix(g, ccs, 2; clear_edges=true)
  @test left_map == [3]
  @test right_map == [1]
  @test Matrix(W) == [4.0;;]
  @test isempty(g.edges_from_left[3])
  @test !isempty(g.edges_from_left[1])
  @test !isempty(g.edges_from_left[2])
end

@testset "BipartiteGraph" begin
  test_get_connected_components(4, 4, 2)
  test_get_connected_components(10, 10, 4)
  test_get_connected_components(187, 294, 18)
  test_get_connected_components(8 * 10^3, 3 * 10^6, 9 * 10^2)
  test_get_cc_matrix()
end
