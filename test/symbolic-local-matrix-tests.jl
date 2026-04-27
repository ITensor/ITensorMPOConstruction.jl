using ITensorMPOConstruction
using ITensorMPOConstruction:
  MPOGraph,
  MPO_symbolic,
  SymbolicMPO,
  SymbolicLocalMatrix,
  _append_symbolic_local_matrix_term!,
  _evaluate_symbolic_local_matrix,
  _internal_symbolic_id,
  _max_user_label,
  _normalize_symbolic_local_matrix!,
  _scale_symbolic_weight,
  _weight_value,
  combine_duplicate_adjacent_right_vertices!,
  internalize_symbolic_ids!,
  left_size,
  left_vertex,
  prepare_opID_sum!
using ITensorMPOConstruction: right_size, terms_eq_from
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
  @test_throws ArgumentError _internal_symbolic_id(-1)

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

function test_internalize_symbolic_ids()::Nothing
  sites = siteinds("Qubit", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X"], ["I", "Z"]])

  X(n) = OpID(2, n)
  Z(n) = OpID(2, n)

  modify!(ops) = 1
  os = OpIDSum{2,Int,Int}(3, op_cache_vec, modify!; abs_tol=0.25)
  add!(os, 1, X(1))
  add!(os, 2, Z(2))
  add!(os, 3, X(1), Z(2))

  original_op_cache_vec = os.op_cache_vec
  original_abs_tol = os.abs_tol
  original_modify! = os.modify!
  original_scalars = os.scalars

  @test internalize_symbolic_ids!(os) === os
  @test original_scalars === os.scalars
  @test os.scalars[1:3] == [2, 3, 4]
  @test os.op_cache_vec === original_op_cache_vec
  @test os.abs_tol == original_abs_tol
  @test os.modify! === original_modify!

  zero_label_os = OpIDSum{2,Int,Int}(1, op_cache_vec)
  add!(zero_label_os, 1, X(1))
  zero_label_os.scalars[1] = 0
  @test_throws ArgumentError internalize_symbolic_ids!(zero_label_os)

  negative_label_os = OpIDSum{2,Int,Int}(1, op_cache_vec)
  add!(negative_label_os, -1, X(1))
  @test_throws ArgumentError internalize_symbolic_ids!(negative_label_os)

  return nothing
end

function symbolic_basis_rewrite_cache_vecs(factor)
  identity_matrix = [1.0 0.0; 0.0 1.0]
  x_matrix = [0.0 1.0; 1.0 0.0]

  op_cache_vec = [[
    OpInfo("I", identity_matrix, false, QN()),
    OpInfo("scaled I", factor * identity_matrix, false, QN()),
    OpInfo("X", x_matrix, false, QN()),
  ]]
  basis_op_cache_vec = [[
    OpInfo("I", identity_matrix, false, QN()), OpInfo("X", x_matrix, false, QN())
  ]]

  return op_cache_vec, basis_op_cache_vec
end

function symbolic_rewrite_opID_sum(factor, user_label::Int=7)
  op_cache_vec, basis_op_cache_vec = symbolic_basis_rewrite_cache_vecs(factor)

  scaled_identity(n) = OpID(2, n)
  X(n) = OpID(3, n)

  os = OpIDSum{2,Int,Int}(1, op_cache_vec)
  add!(os, user_label, scaled_identity(1), X(1))
  internalize_symbolic_ids!(os)

  return os, basis_op_cache_vec
end

function test_symbolic_basis_rewrite_sign_factors()::Nothing
  user_label = 7
  internal_id = user_label + 1

  os, basis_op_cache_vec = symbolic_rewrite_opID_sum(-1.0, user_label)
  prepare_opID_sum!(os, basis_op_cache_vec; symbolic_coefficients=true)

  scalar, ops = os[1]
  nonzero_ops = [op for op in ops if op != zero(op)]
  @test scalar == -internal_id
  @test nonzero_ops == [OpID(2, 1)]
  @test count(op -> op == zero(op), ops) == 1
  @test os.op_cache_vec === basis_op_cache_vec

  os, basis_op_cache_vec = symbolic_rewrite_opID_sum(1.0, user_label)
  prepare_opID_sum!(os, basis_op_cache_vec; symbolic_coefficients=true)

  scalar, ops = os[1]
  nonzero_ops = [op for op in ops if op != zero(op)]
  @test scalar == internal_id
  @test nonzero_ops == [OpID(2, 1)]
  @test count(op -> op == zero(op), ops) == 1
  @test os.op_cache_vec === basis_op_cache_vec

  return nothing
end

function test_symbolic_basis_rewrite_rejects_unsupported_factors()::Nothing
  for factor in (im, 2im, 0.5, 2.0)
    os, basis_op_cache_vec = symbolic_rewrite_opID_sum(factor)
    err = nothing
    try
      prepare_opID_sum!(os, basis_op_cache_vec; symbolic_coefficients=true)
    catch e
      err = e
    end

    @test !isnothing(err)
    error_message = sprint(showerror, err)
    @test occursin("ArgumentError", error_message)
    @test occursin("Unsupported symbolic rewrite factor", error_message)
  end

  return nothing
end

function two_site_qubit_symbolic_fixture()
  sites = siteinds("Qubit", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X"] for _ in eachindex(sites)])
  return sites, op_cache_vec
end

function symbolic_terms_from_first_left_vertex(g)::SymbolicLocalMatrix{Int}
  @test left_size(g) == 1
  lv = left_vertex(g, 1)
  signed_local_op_id = lv.needs_JW_string ? -lv.op_id : lv.op_id
  terms = SymbolicLocalMatrix{Int}()
  for signed_weight_id in g.edge_weights_from_left[1]
    _append_symbolic_local_matrix_term!(terms, signed_weight_id, signed_local_op_id)
  end
  return _normalize_symbolic_local_matrix!(terms)
end

function compact_duplicate_symbolic_right_vertices!(g)::Nothing
  combine_duplicate_adjacent_right_vertices!(g, terms_eq_from(2))
  @test right_size(g) == 1
  @test g.right_vertex_ids_from_left == [[1 for _ in g.edge_weights_from_left[1]]]
  return nothing
end

function test_numeric_mpo_graph_still_compacts_duplicate_terms()::Nothing
  sites = siteinds("Qubit", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X", "Z"] for _ in eachindex(sites)])

  X(n) = OpID(2, n)
  Z(n) = OpID(3, n)

  os = OpIDSum{2,Float64,Int}(4, op_cache_vec; abs_tol=0.25)
  add!(os, 1.0, X(1), X(2))
  add!(os, 2.0, X(1), X(2))
  add!(os, 0.5, Z(1), Z(2))
  add!(os, -0.4, Z(1), Z(2))

  g = MPOGraph(os)

  @test left_size(g) == 1
  @test right_size(g) == 1
  @test left_vertex(g, 1).op_id == 2
  @test g.right_vertex_ids_from_left == [[1]]
  @test g.edge_weights_from_left == [[3.0]]

  return nothing
end

function test_symbolic_mpo_graph_preserves_duplicate_signed_ids()::Nothing
  _, op_cache_vec = two_site_qubit_symbolic_fixture()

  X(n) = OpID(2, n)

  os = OpIDSum{2,Int,Int}(2, op_cache_vec)
  add!(os, 1, X(1), X(2))
  add!(os, 2, X(1), X(2))
  internalize_symbolic_ids!(os)

  g = MPOGraph(os; symbolic_coefficients=true)
  @test length(g.edge_weights_from_left[1]) == 2
  @test sort(g.edge_weights_from_left[1]) == [2, 3]

  compact_duplicate_symbolic_right_vertices!(g)
  terms = symbolic_terms_from_first_left_vertex(g)
  @test terms == [(2, 2), (3, 2)]

  coefficients = [11.0, 17.0]
  evaluated = _evaluate_symbolic_local_matrix(terms, coefficients, op_cache_vec[1])
  @test evaluated ≈ sum(coefficients) * op_cache_vec[1][2].matrix

  return nothing
end

function test_symbolic_mpo_graph_preserves_duplicate_label_multiplicity()::Nothing
  _, op_cache_vec = two_site_qubit_symbolic_fixture()

  X(n) = OpID(2, n)

  os = OpIDSum{2,Int,Int}(2, op_cache_vec)
  add!(os, 1, X(1), X(2))
  add!(os, 1, X(1), X(2))
  internalize_symbolic_ids!(os)

  g = MPOGraph(os; symbolic_coefficients=true)
  compact_duplicate_symbolic_right_vertices!(g)
  terms = symbolic_terms_from_first_left_vertex(g)
  @test terms == [(2, 2), (2, 2)]

  coefficient = 13.0
  evaluated = _evaluate_symbolic_local_matrix(terms, [coefficient], op_cache_vec[1])
  @test evaluated ≈ 2 * coefficient * op_cache_vec[1][2].matrix

  return nothing
end

function signed_symbolic_rewrite_cache_vecs()
  identity_matrix = [1.0 0.0; 0.0 1.0]
  x_matrix = [0.0 1.0; 1.0 0.0]

  op_cache_vec = [
    [
      OpInfo("I", identity_matrix, false, QN()),
      OpInfo("X", x_matrix, false, QN()),
      OpInfo("-X", -x_matrix, false, QN()),
    ],
    [OpInfo("I", identity_matrix, false, QN()), OpInfo("X", x_matrix, false, QN())],
  ]
  basis_op_cache_vec = [
    [OpInfo("I", identity_matrix, false, QN()), OpInfo("X", x_matrix, false, QN())],
    [OpInfo("I", identity_matrix, false, QN()), OpInfo("X", x_matrix, false, QN())],
  ]

  return op_cache_vec, basis_op_cache_vec
end

function test_symbolic_mpo_graph_cancels_opposite_signed_ids()::Nothing
  op_cache_vec, basis_op_cache_vec = signed_symbolic_rewrite_cache_vecs()

  X(n) = OpID(2, n)
  negX(n) = OpID(3, n)

  os = OpIDSum{2,Int,Int}(2, op_cache_vec)
  add!(os, 1, X(1), X(2))
  add!(os, 1, negX(1), X(2))
  internalize_symbolic_ids!(os)
  prepare_opID_sum!(os, basis_op_cache_vec; symbolic_coefficients=true)

  g = MPOGraph(os; symbolic_coefficients=true)
  compact_duplicate_symbolic_right_vertices!(g)
  terms = symbolic_terms_from_first_left_vertex(g)
  @test isempty(terms)

  evaluated = _evaluate_symbolic_local_matrix(terms, [7.0], basis_op_cache_vec[1])
  @test iszero(evaluated)

  return nothing
end

function transverse_field_ising_symbolic_fixture()
  sites = siteinds("Qubit", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X", "Z"] for _ in eachindex(sites)])

  X(n) = OpID(2, n)
  Z(n) = OpID(3, n)

  os = OpIDSum{2,Int,Int}(3, op_cache_vec)
  add!(os, 1, X(1), X(2))
  add!(os, 2, Z(1))
  add!(os, 3, Z(2))

  return os, sites
end

function transverse_field_ising_numeric_opidsum(sites, coefficients)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X", "Z"] for _ in eachindex(sites)])

  X(n) = OpID(2, n)
  Z(n) = OpID(3, n)

  os = OpIDSum{2,eltype(coefficients),Int}(3, op_cache_vec)
  add!(os, coefficients[1], X(1), X(2))
  add!(os, coefficients[2], Z(1))
  add!(os, coefficients[3], Z(2))

  return os
end

function qn_number_symbolic_fixture()
  sites = siteinds("Fermion", 2; conserve_qns=true)
  op_cache_vec = to_OpCacheVec(sites, [["I", "N"] for _ in eachindex(sites)])

  Nop(n) = OpID(2, n)

  os = OpIDSum{2,Int,Int}(3, op_cache_vec)
  add!(os, 1, Nop(1))
  add!(os, 2, Nop(2))
  add!(os, 3, Nop(1), Nop(2))

  return os, sites
end

function qn_number_numeric_opidsum(sites, coefficients)
  op_cache_vec = to_OpCacheVec(sites, [["I", "N"] for _ in eachindex(sites)])

  Nop(n) = OpID(2, n)

  os = OpIDSum{2,eltype(coefficients),Int}(3, op_cache_vec)
  add!(os, coefficients[1], Nop(1))
  add!(os, coefficients[2], Nop(2))
  add!(os, coefficients[3], Nop(1), Nop(2))

  return os
end

function mpo_relative_error(A::MPO, B::MPO)::Float64
  AmB = add(A, -B; alg="directsum")
  numerator = real(inner(AmB, AmB))
  denominator = real(inner(A, A))
  return iszero(denominator) ? numerator : numerator / denominator
end

function test_mpos_approx_equal(A::MPO, B::MPO; tol::Real=1e-10)::Nothing
  relative_error = mpo_relative_error(A, B)
  @test relative_error < tol
  return nothing
end

function symbolic_mpo_terms(sym::SymbolicMPO)::Vector{Tuple{Int,Int}}
  terms = Tuple{Int,Int}[]
  for site_matrices in sym.block_sparse_matrices
    for matrix_of_component in site_matrices
      for block in matrix_of_component
        for block_terms in values(block)
          append!(terms, block_terms)
        end
      end
    end
  end
  return terms
end

function test_symbolic_mpo_construction_metadata()::Nothing
  os, sites = transverse_field_ising_symbolic_fixture()

  sym = MPO_symbolic(os, sites)

  @test sym isa SymbolicMPO
  @test length(sym.offsets) == length(sites)
  @test length(sym.block_sparse_matrices) == length(sites)
  @test length(sym.llinks) == length(sites) + 1
  @test sym.max_user_label == 3
  @test sym.op_cache_vec === os.op_cache_vec
  @test !isempty(symbolic_mpo_terms(sym))

  return nothing
end

function test_symbolic_mpo_rejects_unsupported_public_inputs()::Nothing
  os, sites = transverse_field_ising_symbolic_fixture()
  @test_throws ArgumentError MPO_symbolic(os, sites; alg="QR")

  noninteger_os = OpIDSum{2,Float64,Int}(1, os.op_cache_vec)
  add!(noninteger_os, 1.0, OpID(2, 1), OpID(2, 2))
  @test_throws ArgumentError MPO_symbolic(noninteger_os, sites)

  op_sum = OpSum()
  op_sum += "X", 1
  @test_throws ArgumentError MPO_symbolic(op_sum, sites)

  return nothing
end

function test_symbolic_mpo_fresh_instantiation_matches_numeric()::Nothing
  os, sites = transverse_field_ising_symbolic_fixture()
  sym = MPO_symbolic(os, sites)

  for coefficients in ([1.25, -0.5, 0.75], [-2.0, 0.25, 1.5])
    symbolic_mpo = instantiate_MPO(sym, coefficients; splitblocks=true)
    numeric_mpo = MPO_new(
      transverse_field_ising_numeric_opidsum(sites, coefficients),
      sites;
      alg="VC",
      splitblocks=true,
      checkflux=false,
    )

    test_mpos_approx_equal(symbolic_mpo, numeric_mpo)
  end

  return nothing
end

function test_symbolic_mpo_duplicate_labels_instantiate_like_numeric()::Nothing
  sites, op_cache_vec = two_site_qubit_symbolic_fixture()

  X(n) = OpID(2, n)

  symbolic_os = OpIDSum{2,Int,Int}(2, op_cache_vec)
  add!(symbolic_os, 1, X(1), X(2))
  add!(symbolic_os, 2, X(1), X(2))

  sym = MPO_symbolic(symbolic_os, sites)
  coefficients = [2.0, 3.0]
  symbolic_mpo = instantiate_MPO(sym, coefficients; splitblocks=true)

  numeric_os = OpIDSum{2,Float64,Int}(2, op_cache_vec)
  add!(numeric_os, coefficients[1], X(1), X(2))
  add!(numeric_os, coefficients[2], X(1), X(2))
  numeric_mpo = MPO_new(numeric_os, sites; alg="VC", splitblocks=true, checkflux=false)

  test_mpos_approx_equal(symbolic_mpo, numeric_mpo)

  return nothing
end

function test_symbolic_mpo_qn_fresh_instantiation()::Nothing
  os, sites = qn_number_symbolic_fixture()
  sym = MPO_symbolic(os, sites)

  coefficients = [0.8, -1.1, 0.3]
  for splitblocks in (false, true)
    symbolic_mpo = instantiate_MPO(sym, coefficients; splitblocks, checkflux=true)
    numeric_mpo = MPO_new(
      qn_number_numeric_opidsum(sites, coefficients),
      sites;
      alg="VC",
      splitblocks,
      checkflux=true,
    )

    test_mpos_approx_equal(symbolic_mpo, numeric_mpo)
  end

  nonzero_mpo = instantiate_MPO(sym, [1.0, 2.0, 3.0]; splitblocks=true)
  zero_mpo = instantiate_MPO(sym, [0.0, 0.0, 0.0]; splitblocks=true)

  @test linkdims(zero_mpo) == linkdims(nonzero_mpo)
  @test_throws ArgumentError instantiate_MPO(sym, [1.0, 2.0])

  return nothing
end

function test_symbolic_mpo_uses_negative_local_op_id_for_jw_terms()::Nothing
  sites = siteinds("Fermion", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "C", "Cdag", "N"] for _ in eachindex(sites)])

  C(n) = OpID(2, n)
  Cdag(n) = OpID(3, n)

  os = OpIDSum{2,Int,Int}(1, op_cache_vec)
  add!(os, 1, Cdag(1), C(2))

  sym = MPO_symbolic(os, sites)
  terms = symbolic_mpo_terms(sym)
  @test any(term -> term[2] < 0, terms)

  return nothing
end

@testset "SymbolicLocalMatrix" begin
  test_symbolic_local_matrix_operations()

  test_internalize_symbolic_ids()

  test_symbolic_basis_rewrite_sign_factors()

  test_symbolic_basis_rewrite_rejects_unsupported_factors()

  test_numeric_mpo_graph_still_compacts_duplicate_terms()

  test_symbolic_mpo_graph_preserves_duplicate_signed_ids()

  test_symbolic_mpo_graph_preserves_duplicate_label_multiplicity()

  test_symbolic_mpo_graph_cancels_opposite_signed_ids()

  test_symbolic_mpo_construction_metadata()

  test_symbolic_mpo_rejects_unsupported_public_inputs()

  test_symbolic_mpo_fresh_instantiation_matches_numeric()

  test_symbolic_mpo_duplicate_labels_instantiate_like_numeric()

  test_symbolic_mpo_qn_fresh_instantiation()

  test_symbolic_mpo_uses_negative_local_op_id_for_jw_terms()
end
