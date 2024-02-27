using ITensorMPOConstruction
using ITensors

function halden_shastry_mpo_from_OpSum(N::Int, J::Real; useITensorsAlg::Bool=false)::MPO
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
    return @time "Constructing MPO" MPO(os, sites)
  else
    return @time "Constructing MPO" MPO_new(os, sites)
  end
end

# function ground_state_energy(N::Int, Mtimes2::Int, J::Real)
#   M = abs(Mtimes2) / 2
#   return J * π^2 * ((N - 1 / N) / 24 + M * ((M^2 - 1) / (3 * N^2) - 1 / 4))
# end

let
  N = 800
  J = 1.0

  H, sites = halden_shastry_mpo_from_OpSum(N, 1.0)
  @show maxlinkdim(H)
end

nothing;
