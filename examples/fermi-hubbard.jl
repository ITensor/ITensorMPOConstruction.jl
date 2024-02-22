using ITensorMPOConstruction
using ITensors

function Fermi_Hubbard_momentum_space(
  N::Int, t::Real=1.0, U::Real=4.0; useITensorsAlg::Bool=false
)::MPO
  ## Create the OpSum as per usual.
  os = OpSum{Float64}()
  for k in 1:N
    epsilon = -2 * t * cospi(2 * k / N)
    os .+= epsilon, "Nup", k
    os .+= epsilon, "Ndn", k
  end

  for p in 1:N
    for q in 1:N
      for k in 1:N
        os .+= U / N, "Cdagup", mod1(p - k, N), "Cdagdn", mod1(q + k, N), "Cdn", q, "Cup", p
      end
    end
  end

  sites = siteinds("Electron", N; conserve_qns=true)

  ## The only additional step required is to provide an operator basis in which to represent the OpSum.
  if useITensorsAlg
    return MPO(os, sites)
  else
    operatorNames = [
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

    opCacheVec = [
      [OpInfo(ITensors.Op(name, n), sites[n]) for name in operatorNames] for
      n in eachindex(sites)
    ]

    return MPO_new(os, sites; basisOpCacheVec=opCacheVec)
  end
end

let
  N = 50

  # Ensure compilation
  Fermi_Hubbard_momentum_space(10)

  println("Constructing the Fermi-Hubbard momentum space MPO for $N sites")
  @time mpo = Fermi_Hubbard_momentum_space(N)
  println("The maximum bond dimension is $(maxlinkdim(mpo))")
end

nothing;
