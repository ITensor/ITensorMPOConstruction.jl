# The Haldane-Shastry Hamiltonian defined on ``N`` spin-half particles is

# ```math
# H = \frac{J \pi^2}{N^2} \sum_{n = 1}^N \sum_{m = n + 1}^N \frac{\mathbf{S}_m \cdot \mathbf{S}_n}{\sin^2 \left( \pi \frac{n - m}{N} \right)} \ .
# ```

try
  using MKL
catch
end

using ITensors, ITensorMPS, ITensorMPOConstruction

function haldane_shastry_mpo(
  N::Int, J::Real, tol::Real; use_ITensors_alg::Bool=false
)::MPO
  os = OpSum()
  for n in 1:N
    for m in (n + 1):N
      factor = J * π^2 / N^2 / sinpi((n - m) / N)^2
      os .+= factor, "Sz", n, "Sz", m
      os .+= factor / 2, "S+", n, "S-", m
      os .+= factor / 2, "S-", n, "S+", m
    end
  end

  sites = siteinds("S=1/2", N; conserve_sz=true)

  if use_ITensors_alg
    return MPO(os, sites; cutoff=tol)
  else
    return MPO_new(os, sites; tol)
  end
end;

# With ``N = 40`` and using the default arguments, ITensorMPOConstruction creates an MPO of bond dimension 62, whereas ITensorMPS creates an MPO of bond dimension 38. However, this does not necessarily mean that ITensorMPS produces a better MPO. Comparing MPOs directly is tricky, but since the Haldane-Shastry Hamiltonian is exactly solvable, we can compare the energies and variances of the ground states obtained with DMRG for each MPO.

using LinearAlgebra;
if Threads.nthreads() > 1
  ITensors.Strided.set_num_threads(1)
  BLAS.set_num_threads(1)
  ITensors.enable_threaded_blocksparse(true)
end

function ground_state_energy(N::Int, J::Real, two_Sz::Int)
  M = (N - abs(two_Sz)) / 2
  return J * π^2 * ((N - 1 / N) / 24 + M * ((M^2 - 1) / (3 * N^2) - 1 / 4))
end

function run_dmrg(
  N::Int, two_Sz::Int, maxdim::Int, use_ITensors_alg::Bool, tol::Real, H_maxdim=nothing
)::Nothing
  alg = use_ITensors_alg ? "ITensorMPS" : "ITensorMPOConstruction"
  println("Constructing the Haldane-Shastry MPO for $N sites using $alg with tol=$tol")

  H = haldane_shastry_mpo(N, 1.0, tol; use_ITensors_alg)

  if !isnothing(H_maxdim)
    println("Truncating H...")
    truncate!(H; maxdim=H_maxdim)
  end

  println("The maximum bond dimension is $(maxlinkdim(H))")
  println("The sparsity is $(ITensorMPOConstruction.sparsity(H))")

  N = length(H)
  sites = noprime(siteinds(first, H))

  @assert abs(two_Sz) <= N
  @assert mod(two_Sz, 2) == mod(N, 2)

  nUp = (two_Sz + N) ÷ 2
  psi = random_mps(sites, [i <= nUp ? "Up" : "Dn" for i in 1:N]; linkdims=128)

  E_gs = ground_state_energy(N, 1, two_Sz)
  println("With N = $N, 2 * S_z = $two_Sz, the exact ground state energy is: $E_gs")

  noise = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 0, 0, 0]
  E, psi = dmrg(H, psi; maxdim, cutoff=noise, nsweeps=length(noise), noise, outputlevel=1)
  energy_error = abs(E - E_gs)
  println("The error in the energy from DMRG is: $energy_error")

  HmE = add(H, -E * MPO(sites, "I"))
  variance = inner(HmE, psi, HmE, psi)
  println("The variance of the DMRG state is: $variance")

  return nothing
end;

# In the ``S_z = 0`` sector, with the MPO from ITensorMPOConstruction we obtain an error of ``\delta = E_\text{DMRG} - E_\text{gs} = 5 \times 10^{-14}`` and a variance of ``\sigma^2 = \braket{\psi_\text{DMRG} | (H - E_\text{DMRG})^2 | \psi_\text{DMRG}} = 2 \times 10^{-12}``, whereas with the MPO from ITensors the error increases to ``\delta = 10^{-8}`` while the variance remains unchanged. The fact that the error in the energy increased while the variance remained constant suggests that ITensorMPS is performing a slightly lossy compression the Hamiltonian.

let N = 40, two_Sz = 0, maxdim = 2048
  for (use_ITensors_alg, tol) in ((true, 1e-15), (false, 1.0))
    run_dmrg(N, two_Sz, maxdim, use_ITensors_alg, tol)
    use_ITensors_alg && println()
  end
end

# ````
# Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPS with tol=1.0e-15
# The maximum bond dimension is 38
# The sparsity is 0.8329203539823009
# With N = 40, 2 * S_z = 0, the exact ground state energy is: -16.50074485807127
# ...
# After sweep 10 energy=-16.500744847095405  maxlinkdim=756 maxerr=1.00E-12 time=57.806
# ...
# After sweep 16 energy=-16.50074484722794  maxlinkdim=2048 maxerr=8.07E-17 time=531.916
# The error in the energy from DMRG is: 1.0843329789622658e-8
# The variance of the DMRG state is: 2.2423223836315496e-12

# Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPOConstruction with tol=1.0
# The maximum bond dimension is 62
# The sparsity is 0.8330303718851776
# ...
# After sweep 10 energy=-16.500744857910348  maxlinkdim=749 maxerr=1.00E-12 time=77.843
# ...
# After sweep 16 energy=-16.500744858071318  maxlinkdim=2048 maxerr=8.09E-17 time=697.327
# The error in the energy from DMRG is: 4.973799150320701e-14
# The variance of the DMRG state is: 1.578865487843124e-12
# ````

# ITensorMPOConstruction is designed to construct exact MPOs (up to numerical precision), nevertheless, we can abuse it to perform approximate MPO construction. By setting `tol = 2E10` we obtain an MPO of bond dimension 38, equal to that produced by TensorMPS. However, using this approximate MPO we obtain poor results, with errors of ``\delta = 10^{-3}`` and ``\sigma^2 = 2 \times 10^{-10}``. The fact that such a high tolerance was required to reduce the bond dimension is a sign that this is not a good way of doing things. Setting `absolute_tol = true` to use a uniform cutoff across QR decompositions does not help either.
#
# Starting with the MPO from ITensorMPOConstruction obtained with the standard `tol = 1` and then truncating down to the bond dimension of the MPO from ITensorMPS yields DMRG errors of ``\delta = 4 \times 10^{-9}`` and ``\sigma^2 = 2 \times 10^{-12}``, better than those obtained with the MPO from ITensorMPS.

let N = 40, two_Sz = 0, maxdim = 2048
  for (use_ITensors_alg, tol) in ((false, 2E10), (false, 1))
    H_maxdim = (tol == 1) ? 38 : nothing
    run_dmrg(N, two_Sz, maxdim, use_ITensors_alg, tol, H_maxdim)
    tol > 1 && println()
  end
end

# ````
# Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPOConstruction with tol=2.0e10
# The maximum bond dimension is 38
# ...
# After sweep 16 energy=-16.501803409056485  maxlinkdim=2048 maxerr=6.78E-13 time=536.205
# The error in the energy from DMRG is: 0.0010585509852170105
# The variance of the DMRG state is: 1.6561602614181087e-10

# Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPOConstruction with tol=1
# Truncating H...
# The maximum bond dimension is 38
# ...
# After sweep 16 energy=-16.50074485397465  maxlinkdim=2048 maxerr=8.07E-17 time=539.011
# The error in the energy from DMRG is: 4.0966199321701424e-9
# The variance of the DMRG state is: 2.0995782589283404e-12
# ````
