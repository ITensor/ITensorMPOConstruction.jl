using ITensorMPOConstruction
using ITensorMPOConstruction: terms_eq_from, get_onsite_op, is_fermionic
using ITensors
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
  I, C, Cdag, N = 1, 2, 3, 4

  op_cache_vec = [
    [OpInfo(ITensors.Op(name, n), sites[n]) for name in operatorNames] for
    n in eachindex(sites)
  ]

  @test !is_fermionic((OpID(I, 1),), 1, op_cache_vec)
  @test is_fermionic((OpID(C, 1),), 1, op_cache_vec)
  @test is_fermionic((OpID(Cdag, 1),), 1, op_cache_vec)
  @test !is_fermionic((OpID(N, 1),), 1, op_cache_vec)

  @test !is_fermionic((OpID(I, 1),), 2, op_cache_vec)
  @test !is_fermionic((OpID(C, 1),), 2, op_cache_vec)
  @test !is_fermionic((OpID(Cdag, 1),), 2, op_cache_vec)
  @test !is_fermionic((OpID(N, 1),), 2, op_cache_vec)


  @test !is_fermionic((OpID(C, 5), OpID(Cdag, 2)), 1, op_cache_vec)
  @test !is_fermionic((OpID(C, 5), OpID(Cdag, 2)), 2, op_cache_vec)

  @test is_fermionic((OpID(C, 5), OpID(Cdag, 2)), 3, op_cache_vec)
  @test is_fermionic((OpID(C, 5), OpID(Cdag, 2)), 4, op_cache_vec)
  @test is_fermionic((OpID(C, 5), OpID(Cdag, 2)), 5, op_cache_vec)

  @test !is_fermionic((OpID(C, 5), OpID(Cdag, 2)), 6, op_cache_vec)
end

@testset "Ops" begin
  test_terms_eq_from()

  test_get_onsite_op()

  test_is_fermionic()
end