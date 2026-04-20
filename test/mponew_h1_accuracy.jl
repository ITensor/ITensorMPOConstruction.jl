using ITensorMPOConstruction
using ITensors
using ITensorMPS: siteinds, productMPS, MPO, OpSum, inner, apply, normalize!
using LinearAlgebra
using Random
using Test

function read_h1_terms(path::AbstractString)
  isfile(path) || error("terms file not found: $path")
  N = 0
  terms = Tuple{Int,Int,Float64}[]

  for ln in eachline(path)
    s = strip(ln)
    isempty(s) && continue

    if startswith(s, "#")
      if startswith(s, "# N=")
        N = parse(Int, strip(s[5:end]))
      end
      continue
    end

    f = split(s)
    length(f) == 3 || error("invalid terms line: $ln")
    p = parse(Int, f[1])
    q = parse(Int, f[2])
    h = parse(Float64, f[3])
    push!(terms, (p, q, h))
    N = max(N, p, q)
  end

  isempty(terms) && error("no terms found in $path")
  return N, terms
end

function dense_h1(N::Int, terms)
  H = zeros(Float64, N, N)
  for (p, q, h) in terms
    H[p, q] += h
  end
  return H
end

function sample_sites(n::Int, k::Int)
  k = max(1, min(k, n))
  idx = sort(unique(round.(Int, collect(range(1, n; length=k)))))
  while length(idx) < k
    for i in 1:n
      (i in idx) && continue
      push!(idx, i)
      length(idx) == k && break
    end
  end
  sort!(idx)
  return idx
end

function make_test_orbitals(N::Int; n_basis::Int=8, n_random::Int=4, seed::Int=1234)
  tests = Vector{Vector{Float64}}()

  for p in sample_sites(N, n_basis)
    phi = zeros(Float64, N)
    phi[p] = 1.0
    push!(tests, phi)
  end

  rng = MersenneTwister(seed)
  for _ in 1:n_random
    phi = randn(rng, N)
    nrm = norm(phi)
    nrm > 0 || (phi[1]=1.0; nrm=1.0)
    phi ./= nrm
    push!(tests, phi)
  end

  return tests
end

function onep_state_fermion(sites, phi::AbstractVector{<:Real}; tol::Real=1e-14)
  N = length(sites)
  length(phi) == N || error("orbital length mismatch")

  phi_n = Float64.(phi)
  phi_n ./= norm(phi_n)

  p0 = argmax(abs.(phi_n))
  st = fill("Emp", N)
  st[p0] = "Occ"
  psi0 = productMPS(sites, st)

  os = OpSum()
  for i in 1:N
    c = phi_n[i]
    abs(c) <= tol && continue
    if i == p0
      os += c, "N", p0
    else
      os += c, "Cdag", i, "C", p0
    end
  end

  psi = apply(MPO(os, sites), psi0; alg="naive", cutoff=0.0, maxdim=typemax(Int))
  normalize!(psi)
  return psi
end

function build_h1_opsum(terms)
  os = OpSum()
  for (p, q, h) in terms
    if p == q
      os += h, "N", p
    else
      os += h, "Cdag", p, "C", q
    end
  end
  return os
end

@testset "MPO_new H1 accuracy regression" begin
  data_path = joinpath(@__DIR__, "data", "h1_mponew_bad_case_terms.txt")
  N, terms = read_h1_terms(data_path)

  H = dense_h1(N, terms)
  @test maximum(abs, H .- transpose(H)) <= 1e-12

  sites = siteinds("Fermion", N; conserve_qns=true, conserve_nf=true)
  H_new = MPO_new(build_h1_opsum(terms), sites; tol=1.0, splitblocks=true) # TODO: remove splitblocks=true after removing warning

  tests = make_test_orbitals(N; n_basis=8, n_random=4, seed=1234)
  max_delta = 0.0
  for phi in tests
    psi = onep_state_fermion(sites, phi)
    E_dense = real(dot(phi, H * phi))
    E_new = real(inner(psi', H_new, psi))
    max_delta = max(max_delta, abs(E_new - E_dense))
  end

  # Before fixing the hard-coded assembly cutoff this was ~3e-7 for this case.
  @test max_delta <= 1e-9
end
