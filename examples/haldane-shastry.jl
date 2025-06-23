using ITensorMPOConstruction
using ITensorMPS
using ITensors

function haldane_shastry_mpo(N::Int, J::Real, tol::Real, absolute_tol::Bool; useITensorsAlg::Bool=false)::MPO
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
  M = (N - twoSz) / 2
  return J * π^2 * ((N - 1 / N) / 24 + M * ((M^2 - 1) / (3 * N^2) - 1 / 4))
end

let
  N = 40
  J = 1.0

  useITensorsAlg = false
  # tol = 1
  absolute_tol = true
  tol = 1.39e-3

  # useITensorsAlg = true
  # absolute_tol = false
  # tol = 1e-15

  maxdim = 2^10
  nsweeps = 10
  noise = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0, 0, 0, 0]

  H = haldane_shastry_mpo(N, J, tol, absolute_tol; useITensorsAlg)
  @show useITensorsAlg, tol, absolute_tol, maxlinkdim(H)
  # # truncate!(H, maxdim=38)
  # # @show "truncated dim", maxlinkdim(H)

  sites = noprime(siteinds(first, H))
  psi = random_mps(sites, [mod(i, 2) == 0 ? "Up" : "Dn" for i in 1:N]; linkdims=maxdim)

  E_gs = ground_state_energy(N, J, 0)
  @show E_gs

  E, psi = dmrg(H, psi; nsweeps, maxdim, noise, cutoff=0)
  @show abs(E - E_gs)
  HmE = add(H, -E * MPO(sites, "I"))

  variance = inner(HmE, psi, HmE, psi)
  @show variance
end

nothing;
