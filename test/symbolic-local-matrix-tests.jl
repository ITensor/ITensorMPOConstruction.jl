using ITensorMPOConstruction
using ITensorMPOConstruction:
  SymbolicLocalMatrix,
  _append_symbolic_local_matrix_term!,
  _evaluate_symbolic_local_matrix,
  _internal_symbolic_id,
  _max_user_label,
  _normalize_symbolic_local_matrix!,
  _scale_symbolic_weight,
  _weight_value
using ITensors
using ITensorMPS
using Test

function test_symbolic_local_matrix_operations()::Nothing
  coefficients = [10.0, 20.0]

  @test _weight_value(1, coefficients) == 1.0
  @test _weight_value(-1, coefficients) == -1.0
  @test _weight_value(3, coefficients) == coefficients[2]
  @test _weight_value(-3, coefficients) == -coefficients[2]
  @test_throws ArgumentError _weight_value(4, coefficients)

  @test _internal_symbolic_id(1) == 2
  @test _internal_symbolic_id(3) == 4
  @test_throws ArgumentError _internal_symbolic_id(0)

  @test _scale_symbolic_weight(3, 1) == 3
  @test _scale_symbolic_weight(3, -1) == -3
  @test _scale_symbolic_weight(-3, -1) == 3
  @test_throws ArgumentError _scale_symbolic_weight(3, 2)
  @test_throws ArgumentError _scale_symbolic_weight(3, im)

  terms = SymbolicLocalMatrix{Int}()
  _append_symbolic_local_matrix_term!(terms, 3, -2)
  @test terms == [(3, -2)]
  @test _max_user_label(SymbolicLocalMatrix{Int}([(1, 1), (-3, 2), (5, -2)])) == 4

  terms = SymbolicLocalMatrix{Int}([(3, 2), (-3, 2)])
  _normalize_symbolic_local_matrix!(terms)
  @test isempty(terms)

  terms = SymbolicLocalMatrix{Int}([(3, 2), (3, 2)])
  _normalize_symbolic_local_matrix!(terms)
  @test terms == [(3, 2), (3, 2)]

  terms = SymbolicLocalMatrix{Int}([(5, 3), (4, 2), (3, 2)])
  _normalize_symbolic_local_matrix!(terms)
  @test terms == [(3, 2), (4, 2), (5, 3)]

  qubit_sites = siteinds("Qubit", 1)
  qubit_op_cache_vec = to_OpCacheVec(qubit_sites, [["I", "X"]])
  duplicate_terms = SymbolicLocalMatrix{Int}([(3, 2), (3, 2)])
  @test _evaluate_symbolic_local_matrix(
    duplicate_terms, coefficients, qubit_op_cache_vec[1]
  ) ≈ 2 * coefficients[2] * qubit_op_cache_vec[1][2].matrix

  fermion_sites = siteinds("Fermion", 1)
  fermion_op_cache_vec = to_OpCacheVec(fermion_sites, [["I", "C", "Cdag", "N"]])
  plain_matrix = _evaluate_symbolic_local_matrix(
    SymbolicLocalMatrix{Int}([(3, 2)]), coefficients, fermion_op_cache_vec[1]
  )
  jw_matrix = _evaluate_symbolic_local_matrix(
    SymbolicLocalMatrix{Int}([(3, -2)]), coefficients, fermion_op_cache_vec[1]
  )

  expected_plain_matrix = coefficients[2] * fermion_op_cache_vec[1][2].matrix
  expected_jw_matrix = coefficients[2] * copy(fermion_op_cache_vec[1][2].matrix)
  expected_jw_matrix[:, 2] .*= -1

  @test plain_matrix ≈ expected_plain_matrix
  @test jw_matrix ≈ expected_jw_matrix

  return nothing
end

@testset "SymbolicLocalMatrix" begin
  test_symbolic_local_matrix_operations()
end
