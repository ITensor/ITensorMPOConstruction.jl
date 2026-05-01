# TODO Rename file to AbstractWeight
abstract type AbstractWeight end

# Abs is used to drop terms in OpIDSum which have small coefficients.
Base.abs(::AbstractWeight) = Inf64

struct SimpleWeight <: AbstractWeight
  id::Int
end

Base.one(::Type{SimpleWeight}) = SimpleWeight(typemax(Int))

function Base.:*(s::SimpleWeight, x)::SimpleWeight
  x == +1 && return s
  x == -1 && return SimpleWeight(-s.id)
  error("SimpleWeight can only be multiplied by ±1, not $x")
end

function substitute_weight(weight::SimpleWeight, coefficients::Vector{ValType})::ValType where {ValType}
  abs_weight_id = abs(weight.id)

  value = one(ValType)
  if abs_weight_id != typemax(Int)
    value = coefficients[abs_weight_id]
  end

  return weight.id > 0 ? value : -value
end