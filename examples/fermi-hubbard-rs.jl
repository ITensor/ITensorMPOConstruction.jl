
# # Fermi-Hubbard Hamiltonian in Real Space
# The one-dimensional Fermi-Hubbard Hamiltonian with periodic boundary conditions on ``N`` sites can be expressed in real space as
# ```math
# \mathcal{H} = -t \sum_{i = 1}^N \sum_{\sigma \in (\uparrow, \downarrow)} \left( c^\dagger_{i, \sigma} c_{i + 1, \sigma} + c^\dagger_{i, \sigma} c_{i - 1, \sigma} \right) + U \sum_{i = 1}^N n_{i, \uparrow} n_{i, \downarrow} \ ,
# ```
# where the periodic boundary conditions enforce that ``c_i = c_{i + N}``. For this Hamiltonian, all that needs to be done to switch over to using ITensorMPOConstruction is to switch `MPO(os, sites)` to `MPO_new(os, sites)`.

using ITensors, ITensorMPS, ITensorMPOConstruction

function Fermi_Hubbard_real_space(
  N::Int, t::Real=1.0, U::Real=4.0; use_ITensors_alg::Bool=false
)::MPO
  os = OpSum{Float64}()
  for i in 1:N
    for j in [mod1(i + 1, N), mod1(i - 1, N)]
      os .+= -t, "Cdagup", i, "Cup", j
      os .+= -t, "Cdagdn", i, "Cdn", j
    end

    os .+= U, "Nup * Ndn", i
  end

  sites = siteinds("Electron", N; conserve_qns=true)

  if use_ITensors_alg
    return MPO(os, sites)
  else
    return MPO_new(os, sites)
  end
end;

# For ``N = 1000`` both ITensorMPS and ITensorMPOConstruction can construct an MPO of bond dimension 10 in under two seconds. When quantum number conservation is enabled, ITensorMPS produces an MPO that is 93.4% block sparse, whereas ITensorMPOConstruction's MPO is 97.4% block sparse.

for N in [10, 1000]
  for use_ITensors_alg in [true, false]
    alg = use_ITensors_alg ? "ITensorMPS" : "ITensorMPOConstruction"

    N > 10 &&
      println("Constructing the Fermi-Hubbard real space MPO for $N sites using $alg")
    ITensorMPOConstruction.@time_if (N > 10) 0 "Total construction time" mpo = Fermi_Hubbard_real_space(
      N; use_ITensors_alg
    )

    N > 10 && println("The maximum bond dimension is $(maxlinkdim(mpo))")
    N > 10 && println("The sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
    N > 10 && use_ITensors_alg && println()
  end
end

# ````
# Constructing the Fermi-Hubbard real space MPO for 1000 sites using ITensorMPS
# Total construction time: 1.622434 seconds (23.19 M allocations: 1.937 GiB, 19.34% gc time, 15.05% compilation time)
# The maximum bond dimension is 10
# The sparsity is 0.9343640957766816

# Constructing the Fermi-Hubbard real space MPO for 1000 sites using ITensorMPOConstruction
# Total construction time: 0.553754 seconds (5.22 M allocations: 521.909 MiB, 31.39% gc time, 13.50% compilation time)
# The maximum bond dimension is 10
# The sparsity is 0.9743412345084828
# ````
#
# For a more convincing reason to use `ITensorMPOConstruction` we need to go to momentum space.
