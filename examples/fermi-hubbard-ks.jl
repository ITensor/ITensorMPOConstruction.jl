# # Fermi-Hubbard Hamiltonian in Momentum Space
# The one dimensional Fermi-Hubbard Hamiltonian with periodic boundary conditions on $N$ sites can be expressed in momentum space as
# ```math
# \mathcal{H} = \sum_{k = 1}^N \epsilon(k) \left( 
# n_{k, \downarrow} + n_{k, \uparrow} \right) + \frac{U}{N} \sum_{p, q, k = 1}^N c^\dagger_{p - k, \uparrow} c^\dagger_{q + k, \downarrow} c_{q, \downarrow} c_{p, \uparrow} \ ,
# ```
# where ``\epsilon(k) = -2 t \cos(\frac{2 \pi k}{N})`` and ``c_k = c_{k + N}``.

# Unlike in real space, `MPO_new` is not a drop-in replacement for `MPO`. This is because as expressed above, the Hamiltonian has multiple operators acting on the same site, violating [constraint #3](../documentation/MPO_new.md). For example when ``k = 0`` in the second sum we have terms of the form ``c^\dagger_{p, \uparrow} c^\dagger_{q, \downarrow} c_{q, \downarrow} c_{p, \uparrow}``. 

# You could always create a special case for ``k = 0`` and rewrite it as ``n_{p, \uparrow} n_{q, \downarrow}``. However if using "Electron" sites then you would also need to consider other cases such as when ``p = q``, this would introduce a lot of extraneous complication. Luckily ITensorMPOConstruction provides a method to automatically perform these transformations. If you provide a set of operators foreach site to `MPO_new` it will attempt to express the operators acting on each site as a single one of these "basis" operators.

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

# For ``N = 16`` both ITensorMPS and ITensorMPOConstruction construct an MPO of bond dimension 156. But whereas ITensorMPS takes 6 seconds to produce an MPO of 92% sparsity, ITensorMPOConstruction takes only 0.06 seconds to produce an MPO that is 99.6% sparse.

for N in [4, 16]
  for use_ITensors_alg in [true, false]
    alg = use_ITensors_alg ? "ITensorMPS" : "ITensorMPOConstruction"

    N > 4 && println("Constructing the Fermi-Hubbard real space MPO for $N sites using $alg")
    ITensorMPOConstruction.@time_if (N > 4) 0 "Total construction time" mpo = Fermi_Hubbard_momentum_space(N; use_ITensors_alg)

    N > 4 && println("The maximum bond dimension is $(maxlinkdim(mpo))")
    N > 4 && println("The sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
    N > 4 && use_ITensors_alg && println()
  end
end
