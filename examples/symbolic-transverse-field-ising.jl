# # Symbolic all-to-all transverse-field Ising model
#
# The all-to-all transverse-field Ising model is
# ```math
# H = \sum_{i < j} J_{ij} Z_i Z_j + h \sum_i X_i .
# ```
# If the connectivity changes many times while the operator structure stays
# fixed, the symbolic MPO path can construct the MPO once and instantiate
# it repeatedly for different $J_{ij}$ and $h$.

using ITensors, ITensorMPS, ITensorMPOConstruction
using Random

function num_tfim_couplings(N::Int)::Int
  return (N * (N - 1)) ÷ 2
end

function all_to_all_tfim_symbolic(N::Int)::SymbolicMPO
  sites = siteinds("Qubit", N)
  op_cache_vec = to_OpCacheVec(sites, [["I", "X", "Z"] for _ in 1:N])

  X(n::Int) = OpID(2, n)
  Z(n::Int) = OpID(3, n)

  num_couplings = num_tfim_couplings(N)
  field_label = num_couplings + 1
  os = OpIDSum{2,Int,Int}(num_couplings + N, op_cache_vec)

  label = 1
  for i in 1:N
    for j in (i + 1):N
      add!(os, label, Z(i), Z(j))
      label += 1
    end
  end

  for i in 1:N
    add!(os, field_label, X(i))
  end

  return MPO_symbolic(os, sites)
end

function all_to_all_tfim_numeric(
  sites::Vector{<:Index}, coupling_weights::AbstractVector, h::Real
)::MPO
  N = length(sites)
  num_couplings = num_tfim_couplings(N)

  op_cache_vec = to_OpCacheVec(sites, [["I", "X", "Z"] for _ in 1:N])

  X(n::Int) = OpID(2, n)
  Z(n::Int) = OpID(3, n)

  os = OpIDSum{2,Float64,Int}(num_couplings + N, op_cache_vec)

  label = 1
  for i in 1:N
    for j in (i + 1):N
      add!(os, coupling_weights[label], Z(i), Z(j))
      label += 1
    end
  end

  for i in 1:N
    add!(os, h, X(i))
  end

  return MPO_new(os, sites; alg="VC", splitblocks=true) # TODO: remove splitbocks=true after warning
end

function tfim_symbolic_coefficients(coupling_weights::AbstractVector, h::Real)::Vector{Float64}
  coefficients = Vector{Float64}(undef, length(coupling_weights) + 1)
  coefficients[1:(end - 1)] .= coupling_weights
  coefficients[end] = h
  return coefficients
end

function test_symbolic_tfim_sample(
  sym::SymbolicMPO, coupling_weights::AbstractVector, h::Real
)::Float64
  
  symbolic_mpo = instantiate_MPO(
    sym, tfim_symbolic_coefficients(coupling_weights, h)
  )

  numeric_mpo = all_to_all_tfim_numeric(sym.sites, coupling_weights, h)

  return norm(add(symbolic_mpo, -numeric_mpo; alg="directsum"))
end

let N = 50, h = 1.0, num_samples = 5
  sym = all_to_all_tfim_symbolic(N)
  num_couplings = num_tfim_couplings(N)

  println(
    "Constructed a symbolic all-to-all transverse-field Ising MPO template ",
    "for $N sites",
  )

  for sample in 1:num_samples
    error = test_symbolic_tfim_sample(sym, randn(num_couplings), h)
    println("Sample $sample MPO error: $(round(error; sigdigits=3))")
    @assert iszero(error)
  end
end

nothing
