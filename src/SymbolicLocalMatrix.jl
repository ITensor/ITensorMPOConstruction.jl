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
const SymbolicLocalMatrix{Ti<:Integer} = Vector{Tuple{Int,Ti}}

"""
    SymbolicBlockSparseMatrix{Ti}

Block-sparse symbolic storage mirroring `BlockSparseMatrix`, but with each
block entry represented as a [`SymbolicLocalMatrix`](@ref).
"""
const SymbolicBlockSparseMatrix{Ti<:Integer} =
  Vector{Dictionary{Int,SymbolicLocalMatrix{Ti}}}

function _check_symbolic_weight_id(signed_weight_id::Integer)::Nothing
  signed_weight_id == 0 && throw(
    ArgumentError(
      "Symbolic weight id 0 is invalid. Internal id 1 is reserved for the constant one.",
    ),
  )
  return nothing
end

function _check_symbolic_local_op_id(signed_local_op_id::Integer)::Nothing
  signed_local_op_id == 0 && throw(
    ArgumentError("Symbolic local operator id 0 is invalid. Operator ids are one-based."),
  )
  return nothing
end

function _internal_symbolic_id(user_label::Integer)::Int
  user_label > 0 ||
    throw(ArgumentError("Symbolic coefficient labels must be greater than zero."))
  return user_label + 1
end

"""
    internalize_symbolic_ids!(os::OpIDSum{N,C,Ti}) where {N,C<:Integer,Ti}

Map the integer scalar labels stored in `os` into the internal symbolic id
space used by symbolic MPO construction.

User label `k > 0` is stored as `k + 1`, reserving internal id `1` for the
constant one. Labels less than or equal to zero are invalid. The conversion
mutates `os` in place and does not alter its operator cache, tolerance, or
modification callback.
"""
function internalize_symbolic_ids!(
  os::OpIDSum{N,C,Ti}
)::OpIDSum{N,C,Ti} where {N,C<:Integer,Ti}
  for i in eachindex(os)
    os.scalars[i] = C(_internal_symbolic_id(os.scalars[i]))
  end
  return os
end

function _max_user_label(signed_weight_id::Integer)::Int
  _check_symbolic_weight_id(signed_weight_id)
  return abs(signed_weight_id) - 1
end

function _max_user_label(terms::SymbolicLocalMatrix)::Int
  max_user_label = 0
  for (signed_weight_id, _) in terms
    max_user_label = max(max_user_label, _max_user_label(signed_weight_id))
  end
  return max_user_label
end

function _max_user_label(os::OpIDSum)::Int
  max_user_label = 0
  for i in eachindex(os)
    max_user_label = max(max_user_label, _max_user_label(os.scalars[i]))
  end
  return max_user_label
end

function _append_symbolic_local_matrix_term!(
  terms::SymbolicLocalMatrix{Ti},
  signed_weight_id::Integer,
  signed_local_op_id::Integer,
)::SymbolicLocalMatrix{Ti} where {Ti}
  _check_symbolic_weight_id(signed_weight_id)
  _check_symbolic_local_op_id(signed_local_op_id)
  push!(terms, (Int(signed_weight_id), Ti(signed_local_op_id)))
  return terms
end

function _normalize_symbolic_local_matrix!(
  terms::SymbolicLocalMatrix{Ti}
)::SymbolicLocalMatrix{Ti} where {Ti}
  for (signed_weight_id, signed_local_op_id) in terms
    _check_symbolic_weight_id(signed_weight_id)
    _check_symbolic_local_op_id(signed_local_op_id)
  end

  sort!(terms; by=term -> (term[2], abs(term[1]), term[1]))

  read_index = 1
  write_index = 1
  while read_index <= length(terms)
    signed_local_op_id = terms[read_index][2]
    abs_weight_id = abs(terms[read_index][1])
    num_positive = 0
    num_negative = 0

    while read_index <= length(terms) &&
        terms[read_index][2] == signed_local_op_id &&
        abs(terms[read_index][1]) == abs_weight_id
      if terms[read_index][1] > 0
        num_positive += 1
      else
        num_negative += 1
      end
      read_index += 1
    end

    multiplicity = num_positive - num_negative
    signed_weight_id = multiplicity > 0 ? abs_weight_id : -abs_weight_id
    for _ in 1:abs(multiplicity)
      terms[write_index] = (signed_weight_id, signed_local_op_id)
      write_index += 1
    end
  end

  resize!(terms, write_index - 1)
  return terms
end

function _scale_symbolic_weight(signed_weight_id::Integer, factor)::Int
  _check_symbolic_weight_id(signed_weight_id)
  isone(factor) && return Int(signed_weight_id)
  factor == -one(factor) && return -Int(signed_weight_id)
  throw(
    ArgumentError(
      "Unsupported symbolic rewrite factor $factor. " *
      "Symbolic MPO construction supports only exact +1 and -1 factors.",
    ),
  )
end

function _weight_value(signed_weight_id::Integer, coefficients::AbstractVector)
  _check_symbolic_weight_id(signed_weight_id)
  abs_weight_id = abs(signed_weight_id)
  abs_weight_id == 1 && return signed_weight_id > 0 ? 1.0 : -1.0

  user_label = abs_weight_id - 1
  user_label <= length(coefficients) || throw(
    ArgumentError(
      "Missing coefficient for symbolic label $user_label. " *
      "Received $(length(coefficients)) coefficients.",
    ),
  )

  value = coefficients[user_label]
  return signed_weight_id > 0 ? value : -value
end

function _symbolic_local_matrix_eltype(
  coefficients::AbstractVector, op_cache::Vector{OpInfo}
)::Type
  val_type = promote_type(Float64, eltype(coefficients))
  for op_info in op_cache
    val_type = promote_type(val_type, eltype(op_info.matrix))
  end
  return val_type
end

function _evaluate_symbolic_local_matrix(
  terms::SymbolicLocalMatrix, coefficients::AbstractVector, op_cache::Vector{OpInfo}
)::Matrix
  isempty(op_cache) && throw(
    ArgumentError("Cannot evaluate a symbolic local matrix with an empty operator cache."),
  )

  matrix_size = size(op_cache[1].matrix)
  result = zeros(_symbolic_local_matrix_eltype(coefficients, op_cache), matrix_size)

  for (signed_weight_id, signed_local_op_id) in terms
    _check_symbolic_weight_id(signed_weight_id)
    _check_symbolic_local_op_id(signed_local_op_id)

    local_op_id = abs(Int(signed_local_op_id))
    local_op_id <= length(op_cache) || throw(
      ArgumentError(
        "Symbolic local operator id $local_op_id is not available in a cache with " *
        "$(length(op_cache)) operators.",
      ),
    )

    local_op = op_cache[local_op_id].matrix
    size(local_op) == matrix_size || throw(
      ArgumentError(
        "Operators in a symbolic local matrix must have matching matrix sizes. " *
        "Expected $matrix_size, got $(size(local_op)).",
      ),
    )

    add_to_local_matrix!(
      result, _weight_value(signed_weight_id, coefficients), local_op, signed_local_op_id < 0
    )
  end

  return result
end
