# # Fermi-Hubbard Hamiltonian in Momentum Space
# The one-dimensional Fermi-Hubbard Hamiltonian with periodic boundary conditions on ``N`` sites can be expressed in momentum space as
# ```math
# \mathcal{H} = \sum_{k = 1}^N \epsilon(k) \left(
# n_{k, \downarrow} + n_{k, \uparrow} \right) + \frac{U}{N} \sum_{p, q, k = 1}^N c^\dagger_{p - k, \uparrow} c^\dagger_{q + k, \downarrow} c_{q, \downarrow} c_{p, \uparrow} \ ,
# ```
# where ``\epsilon(k) = -2 t \cos(\frac{2 \pi k}{N})`` and ``c_k = c_{k + N}``.
#
# Unlike in real space, `MPO_new` is not a drop-in replacement for `MPO`. This is because, as expressed above, the Hamiltonian has multiple operators acting on the same site, violating [constraint #3](../documentation/MPO_new.md). For example, when ``k = 0`` in the second sum we have terms of the form ``c^\dagger_{p, \uparrow} c^\dagger_{q, \downarrow} c_{q, \downarrow} c_{p, \uparrow}``.
#
# You could always create a special case for ``k = 0`` and rewrite it as ``n_{p, \uparrow} n_{q, \downarrow}``. However, when using "Electron" sites, you would also need to consider other cases such as ``p = q``, which would introduce a lot of extraneous complication. Luckily, ITensorMPOConstruction provides a method to automatically perform these transformations. If you provide a set of operators on each site to `MPO_new`, it will attempt to express the operators acting on each site as a single one of these "basis" operators.

using ITensors, ITensorMPS, ITensorMPOConstruction

function Fermi_Hubbard_momentum_space(
  N::Int, t::Real=1.0, U::Real=4.0; use_ITensors_alg::Bool=false
)::MPO
  os = OpSum{Float64}()
  for k in 1:N
    epsilon = -2 * t * cospi(2 * k / N)
    os .+= epsilon, "Nup", k
    os .+= epsilon, "Ndn", k
  end

  for p in 1:N
    for q in 1:N
      for k in 1:N
        pmk = mod1(p - k, N)
        qpk = mod1(q + k, N)
        os .+= U / N, "Cdagup", pmk, "Cdagdn", qpk, "Cdn", q, "Cup", p
      end
    end
  end

  sites = siteinds("Electron", N; conserve_qns=true)

  if use_ITensors_alg
    return MPO(os, sites)
  else
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
    operatorNames = [electron_basis_ops for _ in 1:N]

    return MPO_new(os, sites; basis_op_cache_vec=operatorNames)
  end
end;

for N in [4, 16]
  for use_ITensors_alg in [true, false]
    alg = use_ITensors_alg ? "ITensorMPS" : "ITensorMPOConstruction"

    N > 4 &&
      println("Constructing the Fermi-Hubbard momentum space MPO for $N sites using $alg")
    ITensorMPOConstruction.@time_if (N > 4) 0 "Total construction time" mpo = Fermi_Hubbard_momentum_space(
      N; use_ITensors_alg
    )

    N > 4 && println("The maximum bond dimension is $(maxlinkdim(mpo))")
    N > 4 && println("The sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
    N > 4 && use_ITensors_alg && println()
  end
end

# ## Results
# We constructed the 1D momentum-space Fermi-Hubbard Hamiltonian using ITensorMPS and ITensorMPOConstruction with the QR algorithm. For even ``N``, the Hamiltonian can be represented exactly as an MPO of bond dimension ``10 N - 4``, and both methods achieve this minimum bond dimension. However, ITensorMPOConstruction constructs this particular MPO much faster, and the sparsity of the resulting MPO is much higher. This is particularly interesting because the MPO from ITensorMPS was constructed with `splitblocks=true`, whereas for ITensorMPOConstruction we used `splitblocks=false`. The reason for this drastic difference in sparsity is that the momentum-space Fermi-Hubbard Hamiltonian conserves momentum, and even though we did not enforce this symmetry, ITensorMPOConstruction was able to exploit it via the connected-components subroutine. These timings were taken with `julia -t8 --gcthreads=8,1` on a 2021 MacBook Pro with the M1 Max CPU and 32GB of memory. In order to take advantage of the `OpIDSum` machinery, the ITensorMPOConstruction data is from [`fermi-hubbard-tc.jl`](./fermi-hubbard-tc.md) with equivalent settings.
#
# | ``N`` | ITensorMPS    | ITensorMPOConstruction: QR  |
# |-------|---------------|-----------------------------|
# | 10    | 0.32s / 92.7% | 0.01s / 99.32%              |
# | 20    | 30.6s / 92.6% | 0.06s / 99.70%              |
# | 30    | 792s / 92.6%  | 0.53s / 99.81%              |
# | 40    | N/A           | 0.55s / 99.86%              |
# | 50    | N/A           | 0.38s / 99.89%              |
# | 100   | N/A           | 2.81s / 99.94%              |
# | 200   | N/A           | 30.3s / 99.97%              |
# | 300   | N/A           | 136s / 99.982%              |
# | 400   | N/A           | 415s / 99.986%              |
# | 500   | N/A           | 1274s / 99.989%             |
#
# Additionally, we constructed the MPO using the vertex cover algorithm, which turns out to be particularly ill-suited to this problem. Unlike the two other algorithms, the vertex cover algorithm produces MPOs of bond dimension ``1.5 N^2 + 4 N + 2``, far from the optimal of ``10 N - 4``. Granted, with `splitblocks=true` and `N = 100`, the MPO has a sparsity of 99.994%, but with a bond dimension of 15402 it still contains 25 times more nonzero entries than the MPO produced with the QR decomposition.
