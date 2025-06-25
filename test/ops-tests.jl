using ITensorMPOConstruction
using ITensorMPOConstruction: terms_eq_from, get_onsite_op, is_fermionic, sort_fermion_perm!
using ITensors
using ITensorMPS
using Test

function test_terms_eq_from()
  op1 = (OpID(1, 10), OpID(5, 7), OpID(1, 1), OpID(1, 1), OpID(1, 1))
  op2 = (OpID(1, 10), OpID(5, 7), OpID(3, 2), OpID(1, 1), OpID(1, 1))

  @test !terms_eq_from(0)(op1, op2)
  @test !terms_eq_from(1)(op1, op2)
  @test !terms_eq_from(2)(op1, op2)

  @test terms_eq_from(3)(op1, op2)
  @test terms_eq_from(4)(op1, op2)
  @test terms_eq_from(5)(op1, op2)
  @test terms_eq_from(6)(op1, op2)
  @test terms_eq_from(7)(op1, op2)
  @test terms_eq_from(8)(op1, op2)
  @test terms_eq_from(9)(op1, op2)
  @test terms_eq_from(10)(op1, op2)
end

function test_get_onsite_op()
  op = (OpID(1, 10), OpID(5, 7), OpID(1, 1), OpID(1, 1), OpID(1, 1))
  @test get_onsite_op(op, 10) == 1
  @test get_onsite_op(op, 9) == 1
  @test get_onsite_op(op, 8) == 1
  @test get_onsite_op(op, 7) == 5
  @test get_onsite_op(op, 6) == 1
  @test get_onsite_op(op, 5) == 1
  @test get_onsite_op(op, 4) == 1
  @test get_onsite_op(op, 3) == 1
  @test get_onsite_op(op, 2) == 1
  @test get_onsite_op(op, 1) == 1
end

function test_is_fermionic()
  sites = siteinds("Fermion", 10)
  operatorNames = ["I", "C", "Cdag", "N"]
  I(n) = OpID(1, n)
  C(n) = OpID(2, n)
  Cdag(n) = OpID(3, n)
  N(n) = OpID(4, n)

  op_cache_vec = [
    [OpInfo(ITensors.Op(name, n), sites[n]) for name in operatorNames] for
    n in eachindex(sites)
  ]

  @test !is_fermionic((I(1),), 1, op_cache_vec)
  @test is_fermionic((C(1),), 1, op_cache_vec)
  @test is_fermionic((Cdag(1),), 1, op_cache_vec)
  @test !is_fermionic((N(1),), 1, op_cache_vec)

  @test !is_fermionic((I(1),), 2, op_cache_vec)
  @test !is_fermionic((C(1),), 2, op_cache_vec)
  @test !is_fermionic((Cdag(1),), 2, op_cache_vec)
  @test !is_fermionic((N(1),), 2, op_cache_vec)

  @test !is_fermionic((C(5), Cdag(2)), 1, op_cache_vec)
  @test !is_fermionic((C(5), Cdag(2)), 2, op_cache_vec)

  @test is_fermionic((C(5), Cdag(2)), 3, op_cache_vec)
  @test is_fermionic((C(5), Cdag(2)), 4, op_cache_vec)
  @test is_fermionic((C(5), Cdag(2)), 5, op_cache_vec)

  @test !is_fermionic((C(5), Cdag(2)), 6, op_cache_vec)
end

function test_sort_fermion_perm()
  sites = siteinds("Fermion", 10)
  op_cache_vec = to_OpCacheVec(sites, [["I", "C", "Cdag", "N"] for _ in eachindex(sites)])

  I(n) = OpID(1, n)
  C(n) = OpID(2, n)
  Cdag(n) = OpID(3, n)
  N(n) = OpID(4, n)

  ops = [I(4), N(2), N(1)]
  @test sort_fermion_perm!(ops, op_cache_vec) == +1
  @test ops == [N(1), N(2), I(4)]

  ops = [C(2), Cdag(1)]
  @test sort_fermion_perm!(ops, op_cache_vec) == -1
  @test ops == [Cdag(1), C(2)]

  ops = [N(5), C(4), Cdag(4), C(1)]
  @test sort_fermion_perm!(ops, op_cache_vec) == +1
  @test ops == [C(1), C(4), Cdag(4), N(5)]

  ops = [Cdag(1), Cdag(2), C(2), C(1)]
  @test sort_fermion_perm!(ops, op_cache_vec) == +1
  @test ops == [Cdag(1), C(1), Cdag(2), C(2)]
end

@testset "Ops" begin
  test_terms_eq_from()

  test_get_onsite_op()

  test_is_fermionic()

  test_sort_fermion_perm()
end
