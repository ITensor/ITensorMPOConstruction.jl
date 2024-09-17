using ITensorMPOConstruction
using ITensorMPS
using ITensors
using TimerOutputs

function Fermi_Hubbard_real_space(
  N::Int, t::Real=1.0, U::Real=4.0; useITensorsAlg::Bool=false
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

  if useITensorsAlg
    return MPO(os, sites)
  else
    return MPO_new(os, sites)
  end
end

function Fermi_Hubbard_momentum_space(
  N::Int, t::Real=1.0, U::Real=4.0; useITensorsAlg::Bool=false
)::MPO
  os = OpSum{Float64}()

  @time "Constructing OpSum\t" let
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
  end

  sites = siteinds("Electron", N; conserve_qns=true)

  if useITensorsAlg
    return @time "\tConstructing MPO" MPO(os, sites)
  else
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

    return @time "\tConstructing MPO" MPO_new(os, sites; basis_op_cache_vec=operatorNames)
  end
end

function Fermi_Hubbard_momentum_space_OpIDSum(N::Int, t::Real=1.0, U::Real=4.0; output_level::Int=0)::MPO
  sites = siteinds("Electron", N; conserve_qns=true)

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

  op_cache_vec = to_OpCacheVec(sites, operatorNames)

  ↓ = false
  ↑ = true

  opC(k::Int, spin::Bool) = OpID{UInt8}(2 + spin, mod1(k, N))
  opCdag(k::Int, spin::Bool) = OpID{UInt8}(4 + spin, mod1(k, N))
  opN(k::Int, spin::Bool) = OpID{UInt8}(6 + spin, mod1(k, N))

  os = OpIDSum{4, Float64, UInt8}(N^3 + 2 * N, op_cache_vec)

  @time "\tConstructing OpIDSum" let
    for k in 1:N
      epsilon = -2 * t * cospi(2 * k / N)
      add!(os, epsilon, opN(k, ↑))
      add!(os, epsilon, opN(k, ↓))
    end

    for p in 1:N
      for q in 1:N
        for k in 1:N
          add!(os, U / N, opCdag(p - k, ↑), opCdag(q + k, ↓), opC(q, ↓), opC(p, ↑))
        end
      end
    end
  end

  return @time "\tConstructing MPO" MPO_new(
    os, sites; basis_op_cache_vec=op_cache_vec, output_level
  )
end

# let
#   N = 50
#   useITensorsAlg = false

#   println("Constructing the Fermi-Hubbard real space MPO for $N sites")
#   mpo = Fermi_Hubbard_real_space(N; useITensorsAlg=useITensorsAlg)
#   println("The maximum bond dimension is $(maxlinkdim(mpo))")
# end

# let
#   N = 50
#   useITensorsAlg = false

#   println("Constructing the Fermi-Hubbard momentum space MPO for $N sites")
#   mpo = Fermi_Hubbard_momentum_space(N; useITensorsAlg=useITensorsAlg)
#   println("The maximum bond dimension is $(maxlinkdim(mpo))")
# end

let
  N = 10

  println("Constructing the Fermi-Hubbard momentum space MPO for $N sites using OpIDSum")
  mpo = Fermi_Hubbard_momentum_space_OpIDSum(N)
  println("The maximum bond dimension is $(maxlinkdim(mpo))")
  println(ITensorMPOConstruction.sparsity(mpo))
  println()
end

let
  for N in [10, 20, 30, 40, 50, 100, 200]
    println("Constructing the Fermi-Hubbard momentum space MPO for $N sites using OpIDSum")
    @time mpo = Fermi_Hubbard_momentum_space_OpIDSum(N)
    println("The maximum bond dimension is $(maxlinkdim(mpo))")
    println(ITensorMPOConstruction.sparsity(mpo))
    println()
  end
end

nothing;
