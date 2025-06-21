using ITensorMPOConstruction:
  BipartiteGraph,
  compute_connected_components,
  num_connected_components

using Test
using Graphs

function test_get_connected_components(nl::Int, nr::Int, max_edges_from_left::Int)
  g = BipartiteGraph{Int, Int, Float64}(zeros(Int, nl), zeros(Int, nr), [Vector{Tuple{Int, Float64}}() for _ in 1:nl])
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

@testset "BipartiteGraph" begin
  test_get_connected_components(4, 4, 2)
  test_get_connected_components(10, 10, 4)
  test_get_connected_components(187, 294, 18)
  test_get_connected_components(8 * 10^3, 3 * 10^6, 9 * 10^2)
end