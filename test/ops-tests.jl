using ITensorMPOConstruction
using ITensorMPOConstruction: terms_eq_from, get_onsite_op
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

@testset "Ops" begin
  test_terms_eq_from()

  test_get_onsite_op()
end