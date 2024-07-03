struct LeftVertex
  link::Int32
  op_id::Int16
  needs_JW_string::Bool
end

function terms_eq_from(n::Int)::Function
  function are_equal(op1::NTuple{N, OpID}, op2::NTuple{N, OpID})::Bool where {N}
    for i in 1:N
      op1[i].n < n && op2[i].n < n && return true
      op1[i] != op2[i] && return false
    end

    return true
  end

  return are_equal
end

function get_onsite_op(ops::NTuple{N, OpID}, n::Int)::Int16 where {N}
  for i in 1:N
    ops[i].n == n && return ops[i].id
  end

  return 1
end

function is_fermionic(ops::NTuple{N, OpID}, n::Int, op_cache::Vector{OpInfo})::Bool where {N}
  result = false
  for op in ops
    result = xor(result, op.n > n && op_cache[op.id].is_fermionic)
  end

  return result
end
