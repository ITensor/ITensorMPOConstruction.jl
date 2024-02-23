using ITensorMPOConstruction
using ITensors

function electronic_structure_example(
  N::Int, complexBasisFunctions::Bool; useITensorsAlg::Bool=false
)::MPO
  coeff() = rand() + 1im * complexBasisFunctions * rand()

  os = complexBasisFunctions ? OpSum{ComplexF64}() : OpSum{Float64}()
  for a in 1:N
    for b in a:N
      epsilon_ab = coeff()
      os .+= epsilon_ab, "Cdagup", a, "Cup", b
      os .+= epsilon_ab, "Cdagdn", a, "Cdn", b

      a == b && continue
      os .+= conj(epsilon_ab), "Cdagup", b, "Cup", a
      os .+= conj(epsilon_ab), "Cdagdn", b, "Cdn", a
    end
  end

  for j in 1:N
    for s_j in ("dn", "up")
      for k in 1:N
        s_k = s_j
        (s_k, k) <= (s_j, j) && continue

        for l in 1:N
          for s_l in ("dn", "up")
            (s_l, l) <= (s_j, j) && continue

            for m in 1:N
              s_m = s_l
              (s_m, m) <= (s_k, k) && continue

              value = coeff()
              os .+= value, "Cdag$s_j", j, "Cdag$s_l", l, "C$s_m", m, "C$s_k", k
              os .+= conj(value), "Cdag$s_k", k, "Cdag$s_m", m, "C$s_l", l, "C$s_j", j
            end
          end
        end
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
  N = 25

  # Ensure compilation
  electronic_structure_example(5, false)

  println("Constructing the Electronic Structure MPO for $N orbitals")
  @time mpo = electronic_structure_example(N, false)
  println("The maximum bond dimension is $(maxlinkdim(mpo))")
end

nothing;
