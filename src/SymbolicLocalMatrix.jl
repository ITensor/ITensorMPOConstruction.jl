"""
    SymbolicLocalMatrix{Ti}

Symbolic weighted sum of cached local operators for one MPO block entry.

Each term is `(signed_weight_id, signed_local_op_id)`. The signed weight id is
an internal symbolic coefficient id, where `+1` and `-1` represent constants,
and `+k` or `-k` for `k > 1` map to user coefficient `coefficients[k - 1]` with
the stored sign. The absolute value of `signed_local_op_id` is the local
operator id; negative local operator ids indicate that a Jordan-Wigner string
should be applied while evaluating the cached operator matrix.
"""
const SymbolicLocalMatrix{Weight<:AbstractWeight, Ti<:Integer} = Vector{Tuple{Weight, Ti}}

function _append_symbolic_local_matrix_term!(
  terms::SymbolicLocalMatrix{W, Ti},
  weight::W,
  signed_op_id::Ti,
)::Nothing where {W, Ti}
  push!(terms, (weight, signed_op_id))
  return nothing
end

function _evaluate_symbolic_local_matrix!(
  result::Matrix{C}, terms::SymbolicLocalMatrix, coefficients::AbstractVector, op_cache::Vector{OpInfo}
)::Nothing where {C}
  result .= 0

  needs_jw = 0
  for (signed_weight_id, signed_local_op_id) in terms
    local_op_id = abs(Int(signed_local_op_id))
    local_op = op_cache[local_op_id].matrix
    weight = substitute_weight(signed_weight_id, coefficients)

    needs_jw += signed_local_op_id < 0
    add_to_local_matrix!(
      result, weight, local_op, false
    )
  end

  @assert needs_jw ∈ (0, length(terms))
  needs_jw > 0 && apply_jw_string!(result)

  return nothing
end

# TODO: Make sure this isn't used anywhere
function _evaluate_symbolic_local_matrix(
  ::Type{C}, terms::SymbolicLocalMatrix, coefficients::AbstractVector, op_cache::Vector{OpInfo}
)::Matrix{C} where {C}
  result = Matrix{C}(undef, size(op_cache[1].matrix))
  _evaluate_symbolic_local_matrix!(result, terms, coefficients, op_cache)
  return result
end
