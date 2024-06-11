using ITensorMPOConstruction
using ITensors
using ITensorMPS
using Test

function compare_MPOs(A::MPO, B::MPO; tol::Real=1e-7)::Nothing
  diff = add(A, -1 * B; alg="directsum")
  relativeDiff = sqrt(abs(dot(diff, diff))) / norm(B)
  @test relativeDiff < tol
  return nothing
end

function test_from_OpSum(
  os::OpSum, sites::Vector{<:Index}, basisOpCacheVec::Union{Nothing,OpCacheVec}, tol::Real
)::Tuple{MPO,MPO}
  mpo = MPO_new(os, sites; tol=tol, basisOpCacheVec=basisOpCacheVec)

  if tol < 0
    mpoFromITensor = MPO(os, sites)
  else
    mpoFromITensor = MPO(os, sites; cutoff=tol)
  end

  @test all(linkdims(mpo) .<= linkdims(mpoFromITensor))

  compare_MPOs(mpo, mpoFromITensor)

  return mpo, mpoFromITensor
end

function random_complex()::ComplexF64
  return 2 * rand(ComplexF64) - ComplexF64(1, 1)
end

function test_IXYZ(N::Int64, tol::Real)
  β = zeros(ComplexF64, N, 4)
  for i in eachindex(β)
    β[i] = random_complex()
  end

  localOps = ["I", "X", "Y", "Z"]

  os = OpSum{ComplexF64}()
  for string in CartesianIndex(ones(Int64, N)...):CartesianIndex((4 * ones(Int64, N))...)
    string = Tuple(string)

    weight = 1
    args = []
    for i in eachindex(string)
      whichOp = string[i]
      weight *= β[i, whichOp]
      append!(args, (localOps[whichOp], i))
    end

    if weight != 0
      os .+= weight, args...
    end
  end

  sites = siteinds("Qubit", N)
  algMPO, iTensorMPO = test_from_OpSum(os, sites, nothing, tol)

  exact = MPO(sites)

  llinks = Vector{Index}(undef, N + 1)
  llinks[1] = Index(1; tags="Link,l=0")

  for (n, site) in enumerate(sites)
    localOp = β[n, 1] * op(site, "I")
    localOp += β[n, 2] * op(site, "X")
    localOp += β[n, 3] * op(site, "Y")
    localOp += β[n, 4] * op(site, "Z")

    llinks[n + 1] = Index(1; tags="Link,l=$n")
    exact[n] = ITensor(
      ComplexF64, dag(llinks[n]), llinks[n + 1], prime(sites[n]), dag(sites[n])
    )
    exact[n][1, 1, :, :] = array(localOp, prime(sites[n]), dag(sites[n]))
  end

  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(dag(llinks[N + 1]))
  R[1] = 1.0

  exact[1] *= L
  exact[N] *= R

  compare_MPOs(algMPO, exact)
  compare_MPOs(iTensorMPO, exact)

  return nothing
end

function all_ops_of_weight_or_less(N::Integer, weight::Integer)::Vector
  localOps = "X", "Y", "Z"

  if weight == 1
    ops = []
    for i in 1:N
      for op_i in localOps
        push!(ops, [op_i, i])
      end
    end
  else
    ops = all_ops_of_weight_or_less(N, weight - 1)
    for i in eachindex(ops)
      string = ops[i]
      if length(ops[i]) != 2 * (weight - 1)
        continue
      end

      lastSite = string[end]
      for j in (lastSite + 1):N
        for op in localOps
          newOp = copy(string)
          append!(newOp, [op, j])
          push!(ops, newOp)
        end
      end
    end
  end

  return ops
end

function test_random_operator(N::Integer, maxWeight::Integer, tol::Real)::Nothing
  ops = all_ops_of_weight_or_less(N, maxWeight)

  os = OpSum()
  for op in ops
    os .+= random_complex(), op...
  end

  sites = siteinds("Qubit", N)
  test_from_OpSum(os, sites, nothing, tol)

  return nothing
end

function test_Fermi_Hubbard(N::Int, tol::Real)::Nothing
  t, U = 1, 4
  sites = siteinds("Electron", N; conserve_qns=true)

  os = OpSum{Float64}()

  for k in 1:N
    epsilon = cospi(2 * k / N)
    os .+= -t * epsilon, "Nup", k
    os .+= -t * epsilon, "Ndn", k
  end

  for q in 1:N
    for p in 1:N
      for k in 1:N
        os .+= U / N, "Cdagup", mod1(q - k, N), "Cdagdn", mod1(p + k, N), "Cdn", p, "Cup", q
      end
    end
  end

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

  test_from_OpSum(os, sites, opCacheVec, tol)
  return nothing
end

ITensors.op(::OpName"00", ::SiteType"Qubit") = [1 0; 0 0]

ITensors.op(::OpName"01", ::SiteType"Qubit") = [0 1; 0 0]

ITensors.op(::OpName"10", ::SiteType"Qubit") = [0 0; 1 0]

ITensors.op(::OpName"11", ::SiteType"Qubit") = [0 0; 0 1]

function test_qft(N::Int64, applyR::Bool, tol::Real)
  function bits_to_int(bits)
    N = length(bits)
    return sum(bitj << (N - j) for (j, bitj) in enumerate(bits))
  end

  os = OpSum{ComplexF64}()
  for qBits in CartesianIndex(0 * ones(Int64, N)...):CartesianIndex((ones(Int64, N))...)
    for qPrimeBits in
        CartesianIndex(0 * ones(Int64, N)...):CartesianIndex((ones(Int64, N))...)
      qBits = Tuple(qBits)
      qPrimeBits = Tuple(qPrimeBits)

      args = Vector{Op}(undef, N)
      for j in 1:N
        args[j] = Op("$(qBits[j])$(qPrimeBits[j])", j)
      end

      if applyR
        q = bits_to_int(reverse(qBits))
      else
        q = bits_to_int(qBits)
      end

      qPrime = bits_to_int(qPrimeBits)

      weight = 1 / sqrt(2^N) * exp(1im * 2 * π * q * qPrime / 2^N)
      add!(os, weight * Prod(args))
    end
  end

  sites = siteinds("Qubit", N)
  return algMPO, iTensorMPO = test_from_OpSum(os, sites, nothing, tol)
end

@testset "ITensorMPOConstruction.jl" begin
  test_IXYZ(5, -1)

  test_random_operator(8, 4, -1)

  test_qft(6, false, -1)

  test_qft(6, true, -1)

  test_Fermi_Hubbard(12, -1)
end
