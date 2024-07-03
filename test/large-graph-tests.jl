using ITensorMPOConstruction: BipartiteGraph, combine_duplicate_adjacent_right_vertices!, compute_connected_components!
using Test


function test_combine_duplicate_adjacent_right_vertices()::Nothing
  lv = [1, 2, 3]
  rv = [1, 1, 2, 2, 3, 3]
  edges_from_left = [[(1, 1.0), (2, -1.0)], [(3, 2.0), (4, -2.0)], [(5, 3.0), (6, -3.0), (2, 1.0)]]

  g = BipartiteGraph{Int, Int, Float64}(lv, rv, edges_from_left)
  combine_duplicate_adjacent_right_vertices!(g, (a, b) -> a == b)

  return nothing
end

function test_get_connected_components(nl::Int, nr::Int, max_edges_from_left::Int)
  g = BipartiteGraph{Int, Int, Float64}(zeros(Int, nl), zeros(Int, nr), [Vector{Tuple{Int, Float64}}() for _ in 1:nl])

  for lv_id in nl
    for _ in 1:rand(0:max_edges_from_left)
      push!(g.edges_from_left[lv_id], (rand(1:nr), 0.0))
    end
  end

  compute_connected_components!(g)
end

@testset "BipartiteGraph" begin
  test_combine_duplicate_adjacent_right_vertices()

  test_get_connected_components(10^3, 10^3, 4)
end