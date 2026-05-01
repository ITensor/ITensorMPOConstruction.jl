using ITensorMPOConstruction
using ITensorMPOConstruction: MPO_symbolic, SimpleWeight, SymbolicMPO
using ITensors
using ITensorMPS
using Test

function two_site_qubit_symbolic_mpo_fixture()
  sites = siteinds("Qubit", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X"] for _ in eachindex(sites)])
  return sites, op_cache_vec
end

function transverse_field_ising_symbolic_fixture()
  sites = siteinds("Qubit", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X", "Z"] for _ in eachindex(sites)])

  X(n) = OpID(2, n)
  Z(n) = OpID(3, n)

  os = OpIDSum{2,SimpleWeight,Int}(3, op_cache_vec)
  add!(os, SimpleWeight(1), X(1), X(2))
  add!(os, SimpleWeight(2), Z(1))
  add!(os, SimpleWeight(3), Z(2))

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

  os = OpIDSum{2,SimpleWeight,Int}(3, op_cache_vec)
  add!(os, SimpleWeight(1), Nop(1))
  add!(os, SimpleWeight(2), Nop(2))
  add!(os, SimpleWeight(3), Nop(1), Nop(2))

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
  return numerator / denominator
end

function test_mpos_approx_equal(A::MPO, B::MPO; tol::Real=1e-10)::Nothing
  relative_error = mpo_relative_error(A, B)
  @test relative_error < tol
  return nothing
end

function symbolic_mpo_terms(sym::SymbolicMPO)::Vector
  terms = []
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
  @test sym.op_cache_vec === os.op_cache_vec
  @test !isempty(symbolic_mpo_terms(sym))

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
  sites, op_cache_vec = two_site_qubit_symbolic_mpo_fixture()

  X(n) = OpID(2, n)

  symbolic_os = OpIDSum{2,SimpleWeight,Int}(2, op_cache_vec)
  add!(symbolic_os, SimpleWeight(1), X(1), X(2))
  add!(symbolic_os, SimpleWeight(2), X(1), X(2))

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
  @test_throws BoundsError instantiate_MPO(sym, [1.0, 2.0])

  return nothing
end

function test_symbolic_mpo_missing_coefficients_rejected()::Nothing
  os, sites = transverse_field_ising_symbolic_fixture()
  sym = MPO_symbolic(os, sites)
  coefficients = [1.0, 2.0, 3.0]
  short_coefficients = [1.0, 2.0]

  @test_throws BoundsError instantiate_MPO(sym, short_coefficients; splitblocks=true)

  return nothing
end

function test_symbolic_mpo_inplace_instantiation()::Nothing
  os, sites = qn_number_symbolic_fixture()
  sym = MPO_symbolic(os, sites)

  coefficients1 = [0.8, -1.1, 0.3]
  coefficients2 = [-0.2, 1.4, 0.6]

  H = instantiate_MPO(sym, coefficients1; splitblocks=true)
  original_objectid = objectid(H)
  original_linkdims = linkdims(H)

  result = instantiate_MPO!(H, sym, coefficients2; checkflux=true)
  expected = instantiate_MPO(sym, coefficients2; splitblocks=true, checkflux=true)

  @test result === H
  @test objectid(H) == original_objectid
  @test linkdims(H) == original_linkdims
  test_mpos_approx_equal(H, expected)

  coefficients3 = [0.0, 1.4, 0.6]
  result = instantiate_MPO!(H, sym, coefficients3; checkflux=true)
  expected = instantiate_MPO(sym, coefficients3; splitblocks=true, checkflux=true)

  @test result === H
  @test objectid(H) == original_objectid
  @test linkdims(H) == original_linkdims
  test_mpos_approx_equal(H, expected)

  @test_throws MethodError instantiate_MPO!(H, sym, coefficients2; splitblocks=false)

  return nothing
end

function test_symbolic_mpo_template_instantiation()::Nothing
  os, sites = qn_number_symbolic_fixture()
  sym = MPO_symbolic(os, sites)

  coefficients1 = [0.8, -1.1, 0.3]
  coefficients2 = [-0.2, 1.4, 0.6]

  H_template = instantiate_MPO(sym, coefficients1; splitblocks=false)
  H_template_expected = instantiate_MPO(sym, coefficients1; splitblocks=false)
  H = instantiate_MPO(H_template, sym, coefficients2; checkflux=true)
  expected = instantiate_MPO(sym, coefficients2; splitblocks=false, checkflux=true)

  @test H !== H_template
  @test linkdims(H) == linkdims(H_template)
  test_mpos_approx_equal(H, expected)
  test_mpos_approx_equal(H_template, H_template_expected)

  @test_throws MethodError instantiate_MPO(H_template, sym, coefficients2; splitblocks=true)

  return nothing
end

function test_symbolic_mpo_incompatible_template_rejected()::Nothing
  os, sites = qn_number_symbolic_fixture()
  sym = MPO_symbolic(os, sites)
  coefficients = [0.8, -1.1, 0.3]

  bad_os, bad_sites = qn_number_symbolic_fixture()
  bad_sym = MPO_symbolic(bad_os, bad_sites)
  H_bad = instantiate_MPO(bad_sym, coefficients; splitblocks=false)
  @test_throws ArgumentError instantiate_MPO(H_bad, sym, coefficients)

  return nothing
end

function test_symbolic_mpo_uses_negative_local_op_id_for_jw_terms()::Nothing
  sites = siteinds("Fermion", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "C", "Cdag", "N"] for _ in eachindex(sites)])

  C(n) = OpID(2, n)
  Cdag(n) = OpID(3, n)

  os = OpIDSum{2,SimpleWeight,Int}(1, op_cache_vec)
  add!(os, SimpleWeight(1), Cdag(1), C(2))

  sym = MPO_symbolic(os, sites)
  terms = symbolic_mpo_terms(sym)
  @test any(term -> term[2] < 0, terms)

  return nothing
end

@testset "SymbolicMPO" begin
  @testset "fresh instantiation" begin
    test_symbolic_mpo_construction_metadata()
    test_symbolic_mpo_fresh_instantiation_matches_numeric()
    test_symbolic_mpo_duplicate_labels_instantiate_like_numeric()
    test_symbolic_mpo_qn_fresh_instantiation()
    test_symbolic_mpo_uses_negative_local_op_id_for_jw_terms()
  end

  @testset "public API rejections" begin
    test_symbolic_mpo_missing_coefficients_rejected()
    test_symbolic_mpo_incompatible_template_rejected()
  end

  @testset "in-place instantiation" begin
    test_symbolic_mpo_inplace_instantiation()
  end

  @testset "template-assisted instantiation" begin
    test_symbolic_mpo_template_instantiation()
  end
end
