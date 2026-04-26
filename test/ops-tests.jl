using ITensorMPOConstruction
using ITensorMPOConstruction:
  are_equal,
  check_os_for_errors,
  get_onsite_op,
  is_fermionic,
  rewrite_in_operator_basis!,
  sort_fermion_perm!
using ITensors
using ITensorMPS
using Test

function test_are_equal()::Nothing
  op1 = (OpID(1, 10), OpID(5, 7), OpID(1, 1), OpID(1, 1), OpID(1, 1))
  op2 = (OpID(1, 10), OpID(5, 7), OpID(3, 2), OpID(1, 1), OpID(1, 1))

  @test !are_equal(op1, op2, 0)
  @test !are_equal(op1, op2, 1)
  @test !are_equal(op1, op2, 2)

  @test are_equal(op1, op2, 3)
  @test are_equal(op1, op2, 4)
  @test are_equal(op1, op2, 5)
  @test are_equal(op1, op2, 6)
  @test are_equal(op1, op2, 7)
  @test are_equal(op1, op2, 8)
  @test are_equal(op1, op2, 9)
  @test are_equal(op1, op2, 10)

  return nothing
end

function test_get_onsite_op()::Nothing
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

  return nothing
end

function test_is_fermionic()::Nothing
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

  return nothing
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

  return nothing
end

function test_rewrite_in_operator_basis()::Nothing
  sites = siteinds("Qubit", 2)
  op_cache_vec = to_OpCacheVec(sites, [["I", "Z", "X", "Y"] for _ in eachindex(sites)])
  basis_op_cache_vec = to_OpCacheVec(
    sites, [["I", "X", "Y", "Z"] for _ in eachindex(sites)]
  )

  X_orig(n) = OpID(3, n)
  Y_orig(n) = OpID(4, n)

  X_basis(n) = OpID(2, n)
  Z_basis(n) = OpID(4, n)

  os = OpIDSum{3,ComplexF64,Int}(1, op_cache_vec)
  add!(os, 2.0, (X_orig(1), Y_orig(1), X_orig(2)))

  rewrite_in_operator_basis!(os, basis_op_cache_vec)

  scalar, ops = os[1]
  nonzero_ops = [op for op in ops if op != zero(op)]

  @test scalar ≈ 2im
  @test nonzero_ops == [Z_basis(1), X_basis(2)]
  @test count(op -> op == zero(op), ops) == 1
  @test os.op_cache_vec === basis_op_cache_vec

  return nothing
end

function test_rewrite_in_operator_basis_zero()::Nothing
  sites = siteinds("Fermion", 10)
  op_cache_vec = to_OpCacheVec(sites, [["I", "C", "Cdag", "N"] for _ in eachindex(sites)])

  C(n) = OpID(2, n)
  Cdag(n) = OpID(3, n)

  os = OpIDSum{4,Float64,Int}(4, op_cache_vec)
  add!(os, 1, Cdag(1), Cdag(1), C(2), C(3))
  add!(os, -2, C(4), C(4))
  add!(os, 3, Cdag(5), C(6))
  add!(os, 4, C(7), Cdag(8), C(9), C(9))
  rewrite_in_operator_basis!(os, op_cache_vec)

  @test length(os) == 4

  for i in (1, 2, 4)
    scalar, ops = os[i]
    @test iszero(scalar)
    @test all(op -> op == zero(op), ops)
  end

  scalar, ops = os[3]
  nonzero_ops = [op for op in ops if op != zero(op)]
  @test scalar == 3
  @test nonzero_ops == [Cdag(5), C(6)]
  @test os.op_cache_vec === op_cache_vec

  return nothing
end

function test_check_os_for_errors_rejects_linearly_dependent_cache()::Nothing
  sites = siteinds("Qubit", 1)

  let
    op_cache_vec = to_OpCacheVec(sites, [["I", "X", "X"]])
    os = OpIDSum{2,Float64,Int}(0, op_cache_vec)
    @test_throws ErrorException check_os_for_errors(os)
  end

  let
    op_cache_vec = to_OpCacheVec(sites, [["I", "X", "I + X"]])
    os = OpIDSum{2,Float64,Int}(0, op_cache_vec)
    @test_throws ErrorException check_os_for_errors(os)
  end

  return nothing
end

@testset "Ops" begin
  test_are_equal()

  test_get_onsite_op()

  test_is_fermionic()

  test_sort_fermion_perm()

  test_rewrite_in_operator_basis()

  test_rewrite_in_operator_basis_zero()

  test_check_os_for_errors_rejects_linearly_dependent_cache()
end
