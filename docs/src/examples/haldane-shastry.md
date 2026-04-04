The Haldane-Shastry Hamiltonian defined on ``N`` spin-half particles is

```math
H = \frac{J \pi^2}{N^2} \sum_{n = 1}^N \sum_{m = n + 1}^N \frac{\mathbf{S}_m \cdot \mathbf{S}_n}{\sin^2 \left( \pi \frac{n - m}{N} \right)} \ .
```

````julia
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
````

With ``N = 40`` and using the default arguments, ITensorMPOConstruction creates an MPO of bond dimension 62, whereas ITensorMPS creates an MPO of bond dimension 38. However, this does not necessarily mean that ITensorMPS produces a better MPO. Comparing MPOs directly is tricky, but since the Haldane-Shastry Hamiltonian is exactly solvable, we can compare the energies and variances of the ground states obtained with DMRG for each MPO.

````julia
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
  psi = random_mps(sites, [i <= nUp ? "Up" : "Dn" for i in 1:N]; linkdims=maxdim)

  E_gs = ground_state_energy(N, 1, two_Sz)
  println("With N = $N, 2 * S_z = $two_Sz, the exact ground state energy is: $E_gs")

  noise = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 0, 0, 0]
  E, psi = dmrg(H, psi; maxdim, cutoff=noise, nsweeps=length(noise), noise, outputlevel=0)
  energy_error = abs(E - E_gs)
  println("The error in the energy from DMRG is: $energy_error")

  HmE = add(H, -E * MPO(sites, "I"))
  variance = inner(HmE, psi, HmE, psi)
  println("The variance of the DMRG state is: $variance")

  return nothing
end;
````

In the ``S_z = 0`` sector, with the MPO from ITensorMPOConstruction we obtain an error of ``\delta = E_\text{DMRG} - E_\text{gs} = 3 \times 10^{-12}`` and a variance of ``\sigma^2 = \braket{\psi_\text{DMRG} | (H - E_\text{DMRG})^2 | \psi_\text{DMRG}} = 2 \times 10^{-11}``, whereas with the MPO from ITensors the error increases to ``\delta = 10^{-8}`` while the variance remains unchanged. The fact that the error in the energy increased while the variance remained constant suggests that ITensorMPS is performing a slightly lossy compression the Hamiltonian. __Note__: Results shown below with are for ``S_z = 17`` for efficiency when generating these docs, but the implications are the same.

````julia
let N = 40, two_Sz = 34, maxdim = 42
  for (use_ITensors_alg, tol) in ((true, 1e-15), (false, 1.0))
    run_dmrg(N, two_Sz, maxdim, use_ITensors_alg, tol)
    use_ITensors_alg && println()
  end
end
````

````
Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPS with tol=1.0e-15
The maximum bond dimension is 38
The sparsity is 0.8329203539823009
With N = 40, 2 * S_z = 34, the exact ground state energy is: 9.08620455175289
The error in the energy from DMRG is: 8.40004688029694e-9
The variance of the DMRG state is: 1.2979039700994841e-11

Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPOConstruction with tol=1.0
The maximum bond dimension is 62
The sparsity is 0.8330303718851776
With N = 40, 2 * S_z = 34, the exact ground state energy is: 9.08620455175289
The error in the energy from DMRG is: 2.2461676962848287e-10
The variance of the DMRG state is: 4.517458205800995e-11

````

ITensorMPOConstruction is designed to construct exact MPOs (up to numerical precision), nevertheless, we can abuse it to perform approximate MPO construction. By setting `tol = 2E10` we obtain an MPO of bond dimension 38, equal to that produced by TensorMPS. However, using this approximate MPO we obtain poor results, with errors of ``\delta = 10^{-3}`` and ``\sigma^2 = 8 \times 10^{-9}``. The fact that such a high tolerance was required to reduce the bond dimension is a sign that this is not a good way of doing things. Setting `absolute_tol = true` to use a uniform cutoff across QR decompositions does not help either.

Starting with the MPO from ITensorMPOConstruction obtained with the standard `tol = 1` and then truncating down to the bond dimension of the MPO from ITensorMPS yields DMRG errors of ``\delta = 4 \times 10^{-9}`` and ``\sigma^2 = 2 \times 10^{-11}``, better than those obtained with the MPO from ITensorMPS.

````julia
let N = 40, two_Sz = 34, maxdim = 42
  for (use_ITensors_alg, tol) in ((false, 2E10), (false, 1))
    H_maxdim = (tol == 1) ? 38 : nothing
    run_dmrg(N, two_Sz, maxdim, use_ITensors_alg, tol, H_maxdim)
    tol > 1 && println()
  end
end

nothing;
````

````
Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPOConstruction with tol=2.0e10
The maximum bond dimension is 38
The sparsity is 0.8328486610929359
With N = 40, 2 * S_z = 34, the exact ground state energy is: 9.08620455175289
The error in the energy from DMRG is: 0.02071903742048775
The variance of the DMRG state is: 4.856846944572403e-13

Constructing the Haldane-Shastry MPO for 40 sites using ITensorMPOConstruction with tol=1
Truncating H...
The maximum bond dimension is 38
The sparsity is 0.714939197636095
With N = 40, 2 * S_z = 34, the exact ground state energy is: 9.08620455175289
The error in the energy from DMRG is: 3.2155096363339908e-9
The variance of the DMRG state is: 1.061139953191613e-12

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

