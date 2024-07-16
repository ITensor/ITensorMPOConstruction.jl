using ITensorMPOConstruction: BipartiteGraph, combine_duplicate_adjacent_right_vertices!, compute_connected_components!
using Test


function test_combine_duplicate_adjacent_right_vertices()::Nothing
  lv = [1, 2, 3]
  rv = [1, 1, 2, 2, 3, 3]
  edges_from_right = [[(1, 1.0)], [(1, -1.0), (2, 2.0), (3, 1.0)], [(2, -2.0)], [(3, 3.0)], [], [(3, -3.0)]]

  g = BipartiteGraph{Int, Int, Float64}(lv, rv, edges_from_right)
  combine_duplicate_adjacent_right_vertices!(g, (a, b) -> a == b)

  return nothing
end

function test_get_connected_components(nl::Int, nr::Int, max_edges_from_right::Int)
  g = BipartiteGraph{Int, Int, Float64}(zeros(Int, nl), zeros(Int, nr), [Vector{Tuple{Int, Float64}}() for _ in 1:nr])

  for rv_id in nr
    for _ in 1:rand(0:max_edges_from_right)
      push!(g.edges_from_right[rv_id], (rand(1:nl), 0.0))
    end
  end

  compute_connected_components!(g)
end

@testset "BipartiteGraph" begin
  test_combine_duplicate_adjacent_right_vertices()

  test_get_connected_components(10^3, 10^3, 4)
end