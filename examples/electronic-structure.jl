using ITensorMPOConstruction
using ITensorMPS
using ITensors
using PyCall

function get_coefficients(N::Int)::Tuple{Array{Float64,2},Array{Float64,8}}
  py"""
  import numpy as np

  def get_coefficients(N):
    rng = np.random.default_rng(0)
    one_electron = rng.normal(size=(N, N))
    two_electron = rng.normal(size=(N, 2, N, 2, N, 2, N, 2))

    return one_electron, two_electron
  """
  return py"get_coefficients"(N)
end

function electronic_structure(
  N::Int,
  one_electron_coeffs::Array{Float64,2},
  two_electron_coeffs::Array{Float64,8};
  useITensorsAlg::Bool=false,
)::MPO
  os = OpSum{Float64}()
  @time "\tConstructing OpSum" let
    for a in 1:N
      for b in a:N
        epsilon_ab = one_electron_coeffs[a, b]
        os .+= epsilon_ab, "Cdagup", a, "Cup", b
        os .+= epsilon_ab, "Cdagdn", a, "Cdn", b

        a == b && continue
        os .+= conj(epsilon_ab), "Cdagup", b, "Cup", a
        os .+= conj(epsilon_ab), "Cdagdn", b, "Cdn", a
      end
    end

    for j in 1:N
      for s_j in ("dn", "up")
        for k in 1:N
          s_k = s_j
          (s_k, k) <= (s_j, j) && continue

          for l in 1:N
            for s_l in ("dn", "up")
              (s_l, l) <= (s_j, j) && continue

              for m in 1:N
                s_m = s_l
                (s_m, m) <= (s_k, k) && continue

                value = two_electron_coeffs[
                  j,
                  (s_j == "up") + 1,
                  l,
                  (s_l == "up") + 1,
                  m,
                  (s_m == "up") + 1,
                  k,
                  (s_k == "up") + 1,
                ]
                os .+= value, "Cdag$s_j", j, "Cdag$s_l", l, "C$s_m", m, "C$s_k", k
                os .+= conj(value), "Cdag$s_k", k, "Cdag$s_m", m, "C$s_l", l, "C$s_j", j
              end
            end
          end
        end
      end
    end
  end

  sites = siteinds("Electron", N; conserve_qns=true)

  ## The only additional step required is to provide an operator basis in which to represent the OpSum.
  if useITensorsAlg
    return @time "\tConstrucing MPO" MPO(os, sites)
  else
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
    return @time "\tConstrucing MPO" MPO_new(os, sites; basis_op_cache_vec=op_cache_vec)
  end
end

function electronic_structure_OpIDSum(
  N::Int, one_electron_coeffs::Array{Float64,2}, two_electron_coeffs::Array{Float64,8}
)::MPO
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

  ↓ = false
  ↑ = true

  opC(k::Int, spin::Bool) = OpID{UInt8}(2 + spin, k)
  opCdag(k::Int, spin::Bool) = OpID{UInt8}(4 + spin, k)
  opN(k::Int, spin::Bool) = OpID{UInt8}(6 + spin, k)

  os = OpIDSum{4,Float64,UInt8}(2 * N^4, op_cache_vec)
  @time "\tConstructing OpIDSum" let
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

    for j in 1:N
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
  end

  return @time "\tConstructing MPO" MPO_new(os, sites; basis_op_cache_vec=op_cache_vec)
end

let
  for N in [10]
    println("Constructing the electronic structure MPO for $N sites using ITensorMPS")
    one_electron_coeffs, two_electron_coeffs = get_coefficients(N)
    @time "Total construction time" mpo = electronic_structure(
      N, one_electron_coeffs, two_electron_coeffs; useITensorsAlg=true
    )
    println("The maximum bond dimension is $(maxlinkdim(mpo))")
    println("The sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
    println()
  end
end

let
  for N in [10]
    println(
      "Constructing the electronic structure MPO for $N sites using ITensorMPOConstruction"
    )
    one_electron_coeffs, two_electron_coeffs = get_coefficients(N)
    @time "Total construction time" mpo = electronic_structure_OpIDSum(
      N, one_electron_coeffs, two_electron_coeffs
    )
    println("The maximum bond dimension is $(maxlinkdim(mpo))")
    println("The sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
    # @time "splitblocks" mpo = ITensors.splitblocks(linkinds, mpo)
    # println("After splitting the sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
    println()
  end
end

nothing;
