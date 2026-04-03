"""
  LeftVertex

Represents a left vertex of the bipartite MPO graph.
Fields:
- `link`: integer label for the incoming (left) MPO link.
- `op_id`: identifier specifying the operator acting on the current site.
- `needs_JW_string`: whether a Jordan-Wigner string must be inserted when
  connecting through this vertex.
"""
struct LeftVertex
  link::Int32
  op_id::Int16
  needs_JW_string::Bool
end

"""
  are_equal(op1::NTuple{N,OpID}, op2::NTuple{N,OpID}, n::Int) where {N} -> Bool

Check whether two products of OpID are equal from site `n` onward.
The tuples must be sorted by decreasing site.

The comparison proceeds from left to right. If both tuples have already moved to
sites `< n` at the same position, the remaining entries are ignored and the
tuples are considered equal. Otherwise, entries must match exactly.
"""
function are_equal(op1::NTuple{N,OpID}, op2::NTuple{N,OpID}, n::Int)::Bool where {N}
  for i in 1:N
    op1[i].n < n && op2[i].n < n && return true
    op1[i] != op2[i] && return false
  end

  return true
end

"""
  get_onsite_op(ops::NTuple{N,OpID{Ti}}, n::Int) where {N, Ti} -> Ti

Return the operator id in `ops` acting on site `n`.

If no operator acts on site `n`, this returns `1` a.k.a the identity operator.
"""
function get_onsite_op(ops::NTuple{N,OpID{Ti}}, n::Int)::Ti where {N, Ti}
  for i in 1:N
    ops[i].n == n && return ops[i].id
  end

  return 1
end

"""
  is_fermionic(ops::NTuple{N,OpID}, n::Int, op_cache_vec::OpCacheVec) where {N} -> Bool

Compute the fermionic parity of the part of `ops` acting on sites `>= n`.

The result is `true` when an odd number of the operators in `ops` are marked
fermionic in `op_cache_vec`, and `false` otherwise.
"""
function is_fermionic(ops::NTuple{N,OpID}, n::Int, op_cache_vec::OpCacheVec)::Bool where {N}
  result = false
  for op in ops
    result = xor(result, op.n >= n && op_cache_vec[op.n][op.id].is_fermionic)
  end

  return result
end

"""
  ITensors.flux(ops::NTuple{N,OpID}, n::Int, op_cache_vec::OpCacheVec) where {N} -> QN

Return the total quantum-number flux contributed by operators in `ops` on sites
`>= n`, using the cached operator metadata in `op_cache_vec`.
"""
function ITensors.flux(ops::NTuple{N,OpID}, n::Int, op_cache_vec::OpCacheVec)::QN where {N}
  flux = QN()
  for op in ops
    op.n >= n && (flux += op_cache_vec[op.n][op.id].qnFlux)
  end

  return flux
end
