using ITensorMPOConstruction
using ITensorMPS
using ITensors
using Test

function haldane_shastry_mpo(
  N::Int, J::Real, tol::Real, absolute_tol::Bool; useITensorsAlg::Bool=false
)::MPO
  os = OpSum()
  @time "Constructing OpSum" for n in 1:N
    for m in (n + 1):N
      factor = J * π^2 / N^2 / sinpi((n - m) / N)^2
      os .+= factor, "Sz", n, "Sz", m
      os .+= factor / 2, "S+", n, "S-", m
      os .+= factor / 2, "S-", n, "S+", m
    end
  end

  sites = siteinds("S=1/2", N; conserve_sz=true)

  if useITensorsAlg
    return @time "Constructing MPO" MPO(os, sites; cutoff=tol)
  else
    return @time "Constructing MPO" MPO_new(os, sites; tol, absolute_tol)
  end
end

function ground_state_energy(N::Int, J::Real, twoSz::Int)
  M = (N - abs(twoSz)) / 2
  return J * π^2 * ((N - 1 / N) / 24 + M * ((M^2 - 1) / (3 * N^2) - 1 / 4))
end

function test_dmrg(
  H::MPO, twoSz::Int, noise::Vector{Float64}; maxdim::Int=0, compute_variance::Bool=false
)::Nothing
  N = length(H)
  sites = noprime(siteinds(first, H))

  @assert abs(twoSz) <= N
  @assert mod(twoSz, 2) == mod(N, 2)

  maxdim == 0 && (maxdim = 2^(N ÷ 2))
  nUp = (twoSz + N) ÷ 2
  psi = random_mps(sites, [i <= nUp ? "Up" : "Dn" for i in 1:N]; linkdims=maxdim)
  @show flux(psi)

  E_gs = ground_state_energy(N, 1, twoSz)
  println("With N = $N, 2 * S_z = $twoSz, the exact ground state energy is: $E_gs")

  E, psi = dmrg(H, psi; maxdim, cutoff=0, nsweeps=length(noise), noise)
  energy_error = abs(E - E_gs)
  println("The error in the energy from DMRG is: $energy_error")

  @test energy_error < 1e-13

  if compute_variance
    HmE = add(H, -E * MPO(sites, "I"))
    variance = inner(HmE, psi, HmE, psi)
    println("The variance of the DMRG state is: $variance")
  end

  return nothing
end

@testset "HaldaneShastryDMRG" begin
  N = 16
  H = haldane_shastry_mpo(N, 1.0, 1.0, false)

  noise = [1e-5, 1e-6, 1e-7, 0, 0, 0, 0, 0, 0, 0]

  for twoSz in (-N):2:N
    test_dmrg(H, twoSz, noise)
    println()
  end
end

nothing;
