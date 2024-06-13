using ITensorMPOConstruction
using ITensorMPS
using ITensors

function electronic_structure(
  N::Int, complexBasisFunctions::Bool; useITensorsAlg::Bool=false
)::MPO
  coeff() = !complexBasisFunctions ? rand() : rand() + 1im * rand()

  os = complexBasisFunctions ? OpSum{ComplexF64}() : OpSum{Float64}()
  @time "\tConstructing OpSum" let
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
  end

  sites = siteinds("Electron", N; conserve_qns=true)

  ## The only additional step required is to provide an operator basis in which to represent the OpSum.
  if useITensorsAlg
    return @time "\tConstrucing MPO" MPO(os, sites)
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

    op_cache_vec = [
      [OpInfo(ITensors.Op(name, n), sites[n]) for name in operatorNames] for
      n in eachindex(sites)
    ]

    return @time "\tConstrucing MPO" MPO_new(os, sites; basis_op_cache_vec=op_cache_vec)
  end
end

function electronic_structure_OpIDSum(N::Int, complexBasisFunctions::Bool)::MPO
  operatorNames = [
    [
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
    ] for _ in 1:N
  ]

  ↓ = false
  ↑ = true

  opC(k::Int, spin::Bool) = OpID(2 + spin, k)
  opCdag(k::Int, spin::Bool) = OpID(4 + spin, k)
  opN(k::Int, spin::Bool) = OpID(6 + spin, k)

  coeff() = !complexBasisFunctions ? rand() : rand() + 1im * rand()

  os = complexBasisFunctions ? OpIDSum{ComplexF64}() : OpIDSum{Float64}()
  @time "\tConstructing OpIDSum" let
    for a in 1:N
      for b in a:N
        epsilon_ab = coeff()
        push!(os, epsilon_ab, opCdag(a, ↑), opC(b, ↑))
        push!(os, epsilon_ab, opCdag(a, ↓), opC(b, ↓))

        a == b && continue
        push!(os, conj(epsilon_ab), opCdag(b, ↑), opC(a, ↑))
        push!(os, conj(epsilon_ab), opCdag(b, ↓), opC(a, ↓))
      end
    end

    for j in 1:N
      for s_j in (↓, ↑)
        for k in 1:N
          s_k = s_j
          (s_k, k) <= (s_j, j) && continue

          for l in 1:N
            for s_l in (↓, ↑)
              (s_l, l) <= (s_j, j) && continue

              for m in 1:N
                s_m = s_l
                (s_m, m) <= (s_k, k) && continue

                value = coeff()
                push!(os, value, opCdag(j, s_j), opCdag(l, s_l), opC(m, s_m), opC(k, s_k))
                push!(
                  os, conj(value), opCdag(k, s_k), opCdag(m, s_m), opC(l, s_l), opC(j, s_j)
                )
              end
            end
          end
        end
      end
    end
  end

  sites = siteinds("Electron", N; conserve_qns=true)
  return @time "\tConstructing MPO" MPO_new(
    os, sites, operatorNames; basis_op_cache_vec=operatorNames
  )
end

let
  N = 20
  useITensorsAlg = false

  println("Constructing the Electronic Structure MPO for $N orbitals")
  @time "Total" mpo = electronic_structure(N, false; useITensorsAlg=useITensorsAlg)
  println("The maximum bond dimension is $(maxlinkdim(mpo))")
end

let
  N = 20

  println("Constructing the Electronic Structure MPO for $N orbitals using OpIDSum")
  @time "Total" mpo = electronic_structure_OpIDSum(N, false)
  println("The maximum bond dimension is $(maxlinkdim(mpo))")
end

nothing;
