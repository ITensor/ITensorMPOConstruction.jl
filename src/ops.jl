struct LeftVertex
  link::Int32
  op_id::Int16
  needs_JW_string::Bool
end

function are_equal(op1::NTuple{N, OpID}, op2::NTuple{N, OpID}, n::Int)::Bool where {N}
  for i in 1:N
    op1[i].n < n && op2[i].n < n && return true
    op1[i] != op2[i] && return false
  end

  return true
end

function terms_eq_from(n::Int)::Function
  return (op1, op2) -> are_equal(op1, op2, n)
end

function get_onsite_op(ops::NTuple{N, OpID}, n::Int)::Int16 where {N}
  for i in 1:N
    ops[i].n == n && return ops[i].id
  end

  return 1
end

function is_fermionic(ops::NTuple{N, OpID}, n::Int, op_cache_vec::OpCacheVec)::Bool where {N}
  result = false
  for op in ops
    result = xor(result, op.n >= n && op_cache_vec[op.n][op.id].is_fermionic)
  end

  return result
end

function ITensors.flux(ops::NTuple{N, OpID}, n::Int, op_cache_vec::OpCacheVec)::QN where {N}
  flux = QN()
  for op in ops
    op.n >= n && (flux += op_cache_vec[op.n][op.id].qnFlux)
  end
  
  return flux
end