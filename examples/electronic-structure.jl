try
  using MKL
catch
end

using LinearAlgebra
BLAS.set_num_threads(1)

@show Threads.nthreads()
@show BLAS.get_num_threads()

using ITensors, ITensorMPS, ITensorMPOConstruction
using TimerOutputs

function get_coefficients(N::Int)::Tuple{Array{Float64,2},Array{Float64,8}}
  ## If reproducability is required with python
  # using PyCall
  # py"""
  # import numpy as np

  # def get_coefficients(N):
  #   rng = np.random.default_rng(0)
  #   one_electron = rng.normal(size=(N, N))
  #   two_electron = rng.normal(size=(N, 2, N, 2, N, 2, N, 2))

  #   return one_electron, two_electron
  # """
  # return py"get_coefficients"(N)

  return randn(N, N), randn(N, 2, N, 2, N, 2, N, 2)
end

function electronic_structure_OpIDSum(
  N::Int, one_electron_coeffs::Array{Float64,2}, two_electron_coeffs::Array{Float64,8}; electron_sites=true
)::Tuple{Vector{<:Index}, OpIDSum}

  ↓ = false
  ↑ = true

  if electron_sites
    sites = siteinds("Electron", N; conserve_qns=true)

    operatorNames = [
      [
        "I",
        "Cdn",
        "Cup",
        "Cdagdn",
        "Cdagup",
        "Ndn",
        "Nup",
        "Cup * Cdn",
        "Cup * Cdagdn",
        "Cup * Ndn",
        "Cdagup * Cdn",
        "Cdagup * Cdagdn",
        "Cdagup * Ndn",
        "Nup * Cdn",
        "Nup * Cdagdn",
        "Nup * Ndn",
      ] for _ in 1:N
    ]

    op_cache_vec = to_OpCacheVec(sites, operatorNames)
  else
    sites = ITensors.QNIndex[]
    for i in 1:N
      for spin in (-1, +1)
        qns = [
            QN(("Nf", 0, -1), ("Sz",    0)) => 1,
            QN(("Nf", 1, -1), ("Sz", spin)) => 1,
          ]

        push!(sites, Index(qns, "Fermion,Site,$i-$spin"))
      end
    end

    operatorNames = [["I", "C", "Cdag", "N"] for _ in 1:(2 * N)]
    op_cache_vec = to_OpCacheVec(sites, operatorNames)
  end

  opC(k::Int, spin::Bool) = electron_sites ? OpID{UInt8}(2 + spin, k) : OpID{UInt8}(2, 2 * (k - 1) + 1 + spin)
  opCdag(k::Int, spin::Bool) = electron_sites ? OpID{UInt8}(4 + spin, k) : OpID{UInt8}(3, 2 * (k - 1) + 1 + spin)

  os = OpIDSum{4,Float64,UInt8}(2 * N^4, op_cache_vec)
  for a in 1:N
    for b in a:N
      epsilon_ab = one_electron_coeffs[a, b]
      add!(os, epsilon_ab, opCdag(a, ↑), opC(b, ↑))
      add!(os, epsilon_ab, opCdag(a, ↓), opC(b, ↓))

      a == b && continue
      add!(os, conj(epsilon_ab), opCdag(b, ↑), opC(a, ↑))
      add!(os, conj(epsilon_ab), opCdag(b, ↓), opC(a, ↓))
    end
  end

  Threads.@threads for j in 1:N
    for s_j in (↓, ↑)
      for k in 1:N
        s_k = s_j
        (s_k, k) <= (s_j, j) && continue

        for l in 1:N
          for s_l in (↓, ↑)
            (s_l, l) <= (s_j, j) && continue

            for m in 1:N
              s_m = s_l
              (s_m, m) <= (s_k, k) && continue

              value = two_electron_coeffs[j, s_j + 1, l, s_l + 1, m, s_m + 1, k, s_k + 1]
              add!(os, value, opCdag(j, s_j), opCdag(l, s_l), opC(m, s_m), opC(k, s_k))
              add!(
                os, conj(value), opCdag(k, s_k), opCdag(m, s_m), opC(l, s_l), opC(j, s_j)
              )
            end
          end
        end
      end
    end
  end

  return sites, os
end

for N in [10, 30]
  let alg = "VC"
    println("Constructing the electronic structure MPO for $N sites using $alg")

    @time "Constructing OpIDSum" sites, os = electronic_structure_OpIDSum(N, get_coefficients(N)...)

    reset_timer!()
    @time "Constructing MPO" H = MPO_new(os, sites; alg, basis_op_cache_vec=os.op_cache_vec, splitblocks=true, check_for_errors=false, checkflux=false) # TODO: remove splitblocks=true after warning
    N > 10 && print_timer()

    println("The maximum bond dimension is $(maxlinkdim(H)), sparsity = $(ITensorMPOConstruction.sparsity(H))")
    println()
  end
end

nothing;
