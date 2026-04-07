using ITensorMPOConstruction
using ITensorMPS
using ITensors
using TimerOutputs

electron_basis_ops = [
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
]

function transcorrelated_Fermi_Hubbard_momentum_space_OpIDSum(
  N::Int, t::Real=1.0, U::Real=4.0, J::Real=0.0; output_level::Int=0
)::MPO
  sites = siteinds("Electron", N; conserve_qns=true)

  operatorNames = [electron_basis_ops for _ in 1:N]

  op_cache_vec = to_OpCacheVec(sites, operatorNames)

  ↓ = false
  ↑ = true

  opC(k::Int, spin::Bool) = OpID{UInt16}(2 + spin, mod1(k, N))
  opCdag(k::Int, spin::Bool) = OpID{UInt16}(4 + spin, mod1(k, N))
  opN(k::Int, spin::Bool) = OpID{UInt16}(6 + spin, mod1(k, N))

  total_terms = N^3 + 2 * N
  J != 0 && (total_terms += 2 * N^5 + 2 * N^3)
  os = OpIDSum{6,Float64,UInt16}(total_terms, op_cache_vec)

  @time "\tConstructing OpIDSum" let
    epsilon(k) = -2 * t * cospi(2 * k / N)

    for k in 1:N
      add!(os, epsilon(k), opN(k, ↑))
      add!(os, epsilon(k), opN(k, ↓))
    end

    for p in 1:N
      for q in 1:N
        for k in 1:N
          add!(os, U / N, opCdag(p - k, ↑), opCdag(q + k, ↓), opC(q, ↓), opC(p, ↑))
        end
      end
    end

    if J != 0
      for p in 1:N
        for q in 1:N
          for k in 1:N
            omega = ((exp(J) - 1) * epsilon(p - k) + (exp(-J) - 1) * epsilon(p)) / N
            add!(os, omega, opCdag(p - k, ↑), opCdag(q + k, ↓), opC(q, ↓), opC(p, ↑))
            add!(os, omega, opCdag(p - k, ↓), opCdag(q + k, ↑), opC(q, ↑), opC(p, ↓))
          end
        end
      end

      for k in 1:N
        for K in 1:N
          for s in 1:N
            for q in 1:N
              for p in 1:N
                mod(q + K, N) == mod(s + k - K, N) && continue
                s == q && continue

                gamma = 2 * t * (cosh(J) - 1) / (N^2) * epsilon(p - k + K)
                add!(
                  os,
                  gamma,
                  opCdag(p - k, ↑),
                  opCdag(q + K, ↓),
                  opCdag(s + k - K, ↓),
                  opC(s, ↓),
                  opC(q, ↓),
                  opC(p, ↑),
                )
                add!(
                  os,
                  gamma,
                  opCdag(p - k, ↓),
                  opCdag(q + K, ↑),
                  opCdag(s + k - K, ↑),
                  opC(s, ↑),
                  opC(q, ↑),
                  opC(p, ↓),
                )
              end
            end
          end
        end
      end
    end
  end

  return @time "\tConstructing MPO" MPO_new(
    os, sites; basis_op_cache_vec=op_cache_vec, output_level
  )
end

let
  for N in [10, 26]
    println(
      "Constructing the Fermi-Hubbard momentum space MPO for $N sites using ITensorMPOConstruction",
    )
    @time "Total construction time" mpo = transcorrelated_Fermi_Hubbard_momentum_space_OpIDSum(
      N, 1, 4, -0.5
    )
    println("The maximum bond dimension is $(maxlinkdim(mpo))")
    println("The sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
    @show ITensorMPOConstruction.block2_nnz(mpo)
    println()
  end
end

nothing;
