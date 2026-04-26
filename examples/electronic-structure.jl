# # The electronic structure Hamiltonian
#
# With ``N`` spatial orbitals, the electronic structure Hamiltonian is
# ```math
# \mathcal{H} = \sum_{p \leq q = 1}^N \sum_{\sigma \in \{ \uparrow, \downarrow \}} h_{p q} a^\dagger_{p, \sigma} a_{q, \sigma} + \sum_{p, q, r, s}^N \sum_{\sigma, \tau \in \{ \uparrow, \downarrow \}} V_{p q r s} a^\dagger_{p, \sigma} a^\dagger_{q, \tau} a_{r, \tau} a_{s, \sigma}
# ```
# where ``h_{p q} = (p | q)`` and ``V_{p q r s} = (p s | q r)`` in the chemist's notation.
#
# Here we construct an artificial Hamiltonian with random ``h`` and ``V``, though they satisfy the standard permutation symmetries. It turns out that the electronic structure Hamiltonian, even with coefficients from physical systems, is a prime example of when the minimum vertex cover algorithm (`alg = "VC"`) is superior to the QR decomposition algorithm (`alg = "QR"`).

try
  using MKL
catch
end

using LinearAlgebra
BLAS.set_num_threads(1)

@show Threads.nthreads()
@show BLAS.get_num_threads()

using ITensors, ITensorMPS, ITensorMPOConstruction
using Random
using TimerOutputs

"""
    get_coefficients(N) -> Tuple{Array{Float64,2},Array{Float64,4}}

Return random coefficients for the electronic structure Hamiltonian.

The first returned value is the one-electron integral matrix and the second is
the two-electron integral tensor in chemist's notation `V[p, s, q, r] = (ps|qr)`.
"""
function get_coefficients(N::Int)::Tuple{Array{Float64,2},Array{Float64,4}}
  Random.seed!(0)
  h = randn(N, N)
  h = (h + transpose(h)) / 2

  V0 = randn(N, N, N, N)
  V = similar(V0)
  for p in 1:N, s in 1:N, q in 1:N, r in 1:N
    V[p, s, q, r] = (
      V0[p, s, q, r] +
      V0[s, p, q, r] +
      V0[p, s, r, q] +
      V0[s, p, r, q] +
      V0[q, r, p, s] +
      V0[r, q, p, s] +
      V0[q, r, s, p] +
      V0[r, q, s, p]
    ) / 8
  end

  return h, V
end

function electronic_structure_OpIDSum(
  N::Int, h::Array{Float64,2}, V::Array{Float64,4}
)::Tuple{Vector{<:Index}, OpIDSum}

  ↓ = false
  ↑ = true

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

  opC(k::Int, spin::Bool) = OpID{UInt8}(2 + spin, k)
  opCdag(k::Int, spin::Bool) = OpID{UInt8}(4 + spin, k)

  os = OpIDSum{4,Float64,UInt8}(2 * N^4 + 2 * N^2, op_cache_vec)
  for p in 1:N
    for q in 1:N
      for spin in (↓, ↑)
        add!(os, h[p, q], opCdag(p, spin), opC(q, spin))
      end
    end
  end

  Threads.@threads for p in 1:N
    for s_p in (↓, ↑)
      for q in 1:N, s_q in (↓, ↑)
        (q, s_q) <= (p, s_p) && continue

        for r in 1:N, s_r in (↓, ↑)
          for s in 1:N, s_s in (↓, ↑)
            (s, s_s) <= (r, s_r) && continue

            coeff = 0.0
            if s_r == s_q && s_s == s_p
              coeff += V[p, s, q, r]
            end
            if s_s == s_q && s_r == s_p
              coeff -= V[p, r, q, s]
            end
            if s_r == s_p && s_s == s_q
              coeff -= V[q, s, p, r]
            end
            if s_s == s_p && s_r == s_q
              coeff += V[q, r, p, s]
            end

            iszero(coeff) && continue
            add!(os, coeff, opCdag(p, s_p), opCdag(q, s_q), opC(r, s_r), opC(s, s_s))
          end
        end
      end
    end
  end

  return sites, os
end

for alg in ("VC", "QR")
  for N in [10, 20]
    println("Constructing the electronic structure MPO for $N sites using $alg")

    reset_timer!()
    @time "Constructing OpIDSum" sites, os = electronic_structure_OpIDSum(N, get_coefficients(N)...)
    @time "Constructing MPO" H = MPO_new(os, sites; alg, basis_op_cache_vec=os.op_cache_vec, splitblocks=true, check_for_errors=false, checkflux=false) # TODO: remove splitblocks=true after warning
    N > 5 && print_timer()

    println("The maximum bond dimension is $(maxlinkdim(H)), sparsity = $(ITensorMPOConstruction.sparsity(H))")
    @assert maxlinkdim(H) == 2 * N^2 + 3 * N + 2

    println()
  end
end

nothing;

# ## Results
# Below are the runtime and sparsities for the QR decomposition algorithm and the vertex cover algorithm. Both algorithms produce MPOs of bond dimension ``2 N^2 + 3 N + 2``, which is optimal for a generic set of coefficients. These timings were taken with `julia -t8 --gcthreads=8,1` on a 2021 MacBook Pro with the M1 Max CPU and 32GB of memory.
#
# | ``N`` | Minimum Vertex Cover    | QR Decomposition      |
# |-------|-------------------------|-----------------------|
# | 10    | 0.02s / 98.23%          | 0.13s / 95.69%        |
# | 20    | 0.90s / 98.84%          | 1.20s / 97.01%        |
# | 30    | 1.04s / 99.14%          | 8.37s / 97.54%        |
# | 40    | 3.30s / 99.32%          | 16.3s / 97.72%        |
# | 50    | 9.13s / 99.44%          | 56.1s / 97.83%        |
# | 60    | 21.1s / 99.52%          | 169s / 97.90%         |
# | 70    | 50.7s / 99.58%          | 516s / 97.95%         |
# | 80    | 106s / 99.63%           | N/A                   |
# | 90    | 221s / 99.67%           | N/A                   |
# | 100   | 744s / 99.70%           | N/A                   |
#
# Not only does the vertex cover algorithm produce an MPO much faster than the QR algorithm, but the resulting MPO has almost five times fewer entries (note: if `splitblocks=false` then the sparsities of the MPOs are equal). This story remains mostly unchanged when we construct the Hamiltonian for real systems.

# In the table below we present data from constructing two different electronic structure Hamiltonians, the second of which is from [ZhaiLee2023](https://doi.org/10.1021/acs.jpca.3c06142). For the larger system the QR decomposition is able to slightly reduce the bond dimension compared to vertex cover, but it results in a much denser MPO. This increase in density has a significant impact on the subsequent DMRG performance, which takes 70% longer than if the MPO produced by the vertex cover algorithm were used.
#
# | system                                                                     | ``N`` | algorithm | bond dimension | sparsity |
# |----------------------------------------------------------------------------|-------|-----------|----------------|----------|
# | ``\text{C}_2``                                                             | 18    | `QR`      | 704            | 99.26%   |
# | ``\text{C}_2``                                                             | 18    | `VC`      | 704            | 99.77%   |
# | ``\left[\text{Fe}_2 \text{S} (\text{C H}_3)(\text{SCH}_3)_4 \right]^{3-}`` | 36    | `QR`      | 2698           | 97.57%   |
# | ``\left[\text{Fe}_2 \text{S} (\text{C H}_3)(\text{SCH}_3)_4 \right]^{3-}`` | 36    | `VC`      | 2702           | 99.26%   |
#
# ### Thanks to [Huanchen Zhai](https://scholar.google.com/citations?user=HM_YBL0AAAAJ&hl=en) for providing the discussion and data motivating this section.
