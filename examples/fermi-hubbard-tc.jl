# # Challenge Problem: Transcorrelated Fermi-Hubbard
#
# ITensorMPOConstruction was designed specifically to construct the transcorrelated momentum space Fermi-Hubbard Hamiltonian as fast as possible
# ```math
# \begin{aligned}
#   \widetilde{H}_\text{ks}(J) &= H_\text{ks} - \frac{t}{N} \sum_{\bm{p}, \bm{q}, \bm{k}, \sigma} \left[ (e^J - 1) \epsilon(\bm{p} - \bm{k}) + (e^{-J} - 1) \epsilon(\bm{p}) \right] c^\dagger_{\bm{p} - \bm{k}, \sigma} c^\dagger_{\bm{q} + \bm{k}, \bar{\sigma}} c_{\bm{q}, \bar{\sigma}} c_{\bm{p}, \sigma} \\
#   &+ 2 t \frac{\cosh(J) - 1}{N^2} \sum_{\substack{\bm{k}, \bm{k}' \\ \bm{s}, \bm{q} \\ \bm{p}, \sigma}} \epsilon(\bm{p} - \bm{k} + \bm{k}') c^\dagger_{\bm{p} - \bm{k}, \sigma} c^\dagger_{\bm{q} + \bm{k}', \bar{\sigma}} c^\dagger_{\bm{s} + \bm{k} - \bm{k}', \bar{\sigma}} c_{\bm{s}, \bar{\sigma}} c_{\bm{q}, \bar{\sigma}} c_{\mathbf{p}, \sigma} ,
# \end{aligned}
# ```
# where ``J`` is a real number. After ruthless optimization and with a lot of patience we were able to construct the MPO for the ``12 \times 12`` system, which has ``3 \times 10^{10}`` terms. This example goes over the features of ITensorMPOConstruction we designed to make that possible. Additionally, it serves as a reference implementation of the transcorrelated Hamiltonian. For details on the transcorrelated method see our paper: [Scaling up the transcorrelated density matrix renormalization group](https://arxiv.org/abs/2506.07441). 
#
# ## Generating `OpIDSum` directly 
#
# While the Hamiltonian as written above has a number of terms that scales as ``2 N^5``, by grouping like-terms together total number of terms can be reduced to ``N^5 / 2``, nevertheless, simply constructing the `OpIDSum` for this Hamiltonian is a hard problem, for the ``12 \times 12`` system the `OpIDSum` alone takes up almost 600GiB. Because of this we needed to construct the `OpIDSum` directly to avoid constructing the much more costly (in both time and space) `OpSum`. 

try
  using MKL
catch
end

using LinearAlgebra
BLAS.set_num_threads(1)

@show Threads.nthreads()
@show BLAS.get_num_threads()

using ITensors, ITensorMPS, ITensorMPOConstruction

↓ = false
↑ = true

function create_op_cache_vec(sites::Vector{<:Index})::OpCacheVec
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

  return to_OpCacheVec(sites, [operatorNames for _ in 1:length(sites)])
end

function wrap_around(
  grid_size::NTuple{N,Int}, k::CartesianIndex{N}
)::CartesianIndex{N} where {N}
  CartesianIndex(NTuple{N,Int}(mod1(k[i], grid_size[i]) for i in 1:N))
end

function opC(j::CartesianIndex{N}, spin::Bool, mapping::Array{Int,N})::OpID{UInt8} where {N}
  return OpID{UInt8}(2 + spin, mapping[wrap_around(size(mapping), j)])
end

function opCdag(
  j::CartesianIndex{N}, spin::Bool, mapping::Array{Int,N}
)::OpID{UInt8} where {N}
  return OpID{UInt8}(4 + spin, mapping[wrap_around(size(mapping), j)])
end

function opN(j::CartesianIndex{N}, spin::Bool, mapping::Array{Int,N})::OpID{UInt8} where {N}
  return OpID{UInt8}(6 + spin, mapping[wrap_around(size(mapping), j)])
end

function is_spin_dn(op_id)::Bool
  @assert 0 < op_id < 8
  return mod(op_id, 2) == 0
end

# ## Combining local operators
#
# The next bottleneck in construction is combining multiple operators acting on the same site into a single basis operator to satisfy [constraint #3](../documentation/MPO_new.md). This entails for example, replacing ``c^\dagger_{k, \uparrow} c_{k, \uparrow}`` with ``n_{k, \uparrow}``. While this can be accomplished by passing a `basis_op_chache_vec` to `MPO_new`, that code path is slow since it relies on numeric computations to rewrite each product. If instead, we are intelligent about how we formulate the individual terms, we can rewrite them symbolically.
#
# By utilizing the `modify!` function of `OpIDSum` we can let `OpIDSum` do the tricky business of sorting the fermionic operators, and we can then modify the resulting term after operators acting on the same site have been made adjacent. Because the sort is stable, if the original term contains normal ordered spin-up operators followed by normal ordered spin-down operators this structure will be maintained in the operators acting on each site. This allows us to combine the spin-up operators and spin-down operators separately first before merging them together. This does require some changes to the code constructing the `OpIDSum` to respect this ordering in construction.

function merge_sorted_ops(ops::AbstractVector{OpID{UInt8}})::Int
  ITensorMPOConstruction.for_equal_sites(ops) do a, c
    local_ops = view(ops, a:c)
    length(local_ops) == 1 && return nothing

    b = findfirst(is_spin_dn(op.id) for op in local_ops)
    isnothing(b) && (b = length(local_ops) + 1)

    spin_up_ops = view(local_ops, 1:(b - 1))
    if length(spin_up_ops) == 0
      spin_up_op = 1 ## I
    elseif length(spin_up_ops) == 1
      spin_up_op = spin_up_ops[1].id
    else
      @assert length(spin_up_ops) == 2
      @assert spin_up_ops[1].id == 5 ## Cdagup
      @assert spin_up_ops[2].id == 3 ## Cup

      spin_up_op = 7 ## Nup
    end

    spin_dn_ops = view(local_ops, b:length(local_ops))
    @assert all(is_spin_dn(op.id) for op in spin_dn_ops) "$spin_dn_ops"
    if length(spin_dn_ops) == 0
      spin_dn_op = 1
    elseif length(spin_dn_ops) == 1
      spin_dn_op = spin_dn_ops[1].id
    else
      @assert length(spin_dn_ops) == 2
      @assert spin_dn_ops[1].id == 4 ## Cdagdn
      @assert spin_dn_ops[2].id == 2 ## Cdn

      spin_dn_op = 6 ## Ndn
    end

    if spin_up_op == 1 || spin_dn_op == 1
      combined_op = spin_up_op * spin_dn_op
    elseif spin_up_op == 3
      combined_op = 7 + spin_dn_op ÷ 2
    elseif spin_up_op == 5
      combined_op = 10 + spin_dn_op ÷ 2
    else
      @assert spin_up_op == 7
      combined_op = 13 + spin_dn_op ÷ 2
    end

    local_ops[1] = OpID{UInt8}(combined_op, local_ops[1].n)
    for j in 2:length(local_ops)
      local_ops[j] = zero(local_ops[j])
    end
  end

  sort!(ops; by=op -> op.n)
  return 1
end;

# ## General helper code

## Function to construct the momentum conserving site indices
function sites_from_grid(
  mapping::Array{Int}, conserve_momentum::Bool
)::Vector{ITensors.QNIndex}
  !conserve_momentum && return siteinds("Electron", length(mapping); conserve_qns=true)

  spatial_dimension = ndims(mapping)
  @assert 1 <= spatial_dimension <= 2

  indices = Vector{ITensors.QNIndex}(undef, length(mapping))
  for loc in CartesianIndices(mapping)
    if spatial_dimension == 1
      k = only(loc)
      L = length(mapping)
      qns = [
        QN(("Nf", 0, -1), ("Sz", +0), ("Kx", 0, L)) => 1,
        QN(("Nf", 1, -1), ("Sz", +1), ("Kx", k, L)) => 1,
        QN(("Nf", 1, -1), ("Sz", -1), ("Kx", k, L)) => 1,
        QN(("Nf", 2, -1), ("Sz", +0), ("Kx", 2 * k, L)) => 1,
      ]
    else
      kx, ky = Tuple(loc)
      Lx, Ly = size(mapping)
      qns = [
        QN(("Nf", 0, -1), ("Sz", +0), ("Kx", 0, Lx), ("Ky", 0, Ly)) => 1,
        QN(("Nf", 1, -1), ("Sz", +1), ("Kx", kx, Lx), ("Ky", ky, Ly)) => 1,
        QN(("Nf", 1, -1), ("Sz", -1), ("Kx", kx, Lx), ("Ky", ky, Ly)) => 1,
        QN(("Nf", 2, -1), ("Sz", +0), ("Kx", 2 * kx, Lx), ("Ky", 2 * ky, Ly)) => 1,
      ]
    end

    tags = "Electron,Site,$(Tuple(loc))"
    indices[mapping[loc]] = Index(qns, tags)
  end

  return indices
end

## Uses the symmetries of cosine to return numerically equal values when possible.
function my_cospi(x::Rational)
  x = mod(x, 2)
  if x > 1
    x -= 2
  end

  x = abs(x)

  sgn = +1
  if x > 1 // 2
    x = 1 - x
    sgn = -1
  end

  return sgn * cospi(x)
end

function epsilon(gridSize::NTuple{N,Int}, k::CartesianIndex{N})::Float64 where {N}
  return 2 * sum(my_cospi(2 * k[i] // gridSize[i]) for i in 1:N)
end

# ## Constructing the `OpIDSum`
#
# In the function below we construct the `OpIDSum`. There are three things worth highlighting.
#
# 1. The initialization of the `OpIDSum`: We specify the weight of the Hamiltonian, which is 6 in this case, and use a `UInt8` to enumerate the operators and sites, which does limit us to 255 sites. Additionally, we specify the maximum number of terms, our `merge_sorted_ops` function, and a non-zero `abs_tol` to drop terms with coefficients smaller than `1e-14`. This last point is to account for floating point round error for terms whose coefficients are analytically zero.
#
# 2. In order to permit the functioning of `merge_sorted_ops`, we always put spin-up operators before spin-down operators in each term.
#
# 3. Adding the three-electron terms dominate the runtime of this function, but we can add them in parallel with a simple `Threads.@threads`. Thread safety is handled internally to `add!`.

function transcorrelated_fermi_hubbard(
  t::Real, U::Real, J::Real, mapping::Array{Int}; conserve_momentum::Bool=true
)::Tuple{Vector{ITensors.QNIndex},OpIDSum}
  grid_size = size(mapping)
  N = length(mapping)

  sites = sites_from_grid(mapping, conserve_momentum)
  os = OpIDSum{6,Float64,UInt8}(
    (J == 0) ? N^3 + 2 * N : N^5 ÷ 2,
    create_op_cache_vec(sites),
    merge_sorted_ops;
    abs_tol=1e-14,
  )

  ## The hopping term
  for k in CartesianIndices(mapping)
    for sigma in (↓, ↑)
      add!(os, -t * epsilon(grid_size, k), opN(k, sigma, mapping))
    end
  end

  ## The interaction term
  for p in CartesianIndices(mapping)
    for q in CartesianIndices(mapping)
      for k in CartesianIndices(mapping)
        factor = U / N
        ops = opCdag(p - k, ↑, mapping),
        opC(p, ↑, mapping), opCdag(q + k, ↓, mapping),
        opC(q, ↓, mapping)
        add!(os, factor, ops)
      end
    end
  end

  J == 0 && return sites, os

  ## The transcorrelated two electron terms
  for p in CartesianIndices(mapping)
    for q in CartesianIndices(mapping)
      for k in CartesianIndices(mapping)
        factor =
          -t * (
            (exp(J) - 1) * epsilon(grid_size, p - k) + (exp(-J) - 1) * epsilon(grid_size, p)
          ) / N

        ops = opCdag(p - k, ↑, mapping),
        opC(p, ↑, mapping), opCdag(q + k, ↓, mapping),
        opC(q, ↓, mapping)
        add!(os, factor, ops)

        ops = opCdag(q + k, ↑, mapping),
        opC(q, ↑, mapping), opCdag(p - k, ↓, mapping),
        opC(p, ↓, mapping)
        add!(os, factor, ops)
      end
    end
  end

  ## The transcorrelated three electron terms
  Threads.@threads for p in CartesianIndices(mapping)
    for q in CartesianIndices(mapping)
      for s in CartesianIndices(mapping)
        q >= s && continue

        for k in CartesianIndices(mapping)
          for kp in CartesianIndices(mapping)
            wrap_around(grid_size, q + kp) <= wrap_around(grid_size, s + k - kp) && continue

            ## Add up the contributions from the four terms.
            factor =
              2 * t * (cosh(J) - 1) / N^2 * (
                + epsilon(grid_size, p - (k - kp))         ## From (q + kp    , s + k - kp, s, q)
                - epsilon(grid_size, p - (s + k - kp - q)) ## From (q + kp    , s + k - kp, q, s)
                -
                epsilon(grid_size, p - (q + kp - s))     ## From (s + k - kp, q + kp    , s, q)
                + epsilon(grid_size, p - kp)               ## From (s + k - kp, q + kp    , q, s)
              )

            ops = opCdag(p - k, ↑, mapping),
            opC(p, ↑, mapping),
            opCdag(q + kp, ↓, mapping),
            opCdag(s + k - kp, ↓, mapping),
            opC(s, ↓, mapping),
            opC(q, ↓, mapping)
            add!(os, factor, ops)

            ops = opCdag(q + kp, ↑, mapping),
            opCdag(s + k - kp, ↑, mapping),
            opC(s, ↑, mapping),
            opC(q, ↑, mapping),
            opCdag(p - k, ↓, mapping),
            opC(p, ↓, mapping)
            add!(os, factor, ops)
          end
        end
      end
    end
  end

  return sites, os
end

# ## Mappings from momentum space to MPS sites
#
# These have little to do with ITensorMPOConstruction, but are included for completeness. In our paper, we use the "epsilon" mapping for dilute systems, and the "bipartite" mapping at half-filling.

function standard_mapping(grid_size)::Array{Int}
  mapping = zeros(Int, grid_size...)
  for (i, loc) in enumerate(CartesianIndices(mapping))
    mapping[loc] = i
  end

  return mapping
end

function epsilon_mapping(grid_size::Tuple)::Array{Int}
  kAndEps = vec([(k, epsilon(grid_size, k)) for k in each_grid_site(grid_size)])
  sort!(kAndEps; by=(kAndEps) -> kAndEps[2])

  mapping = zeros(Int, grid_size...)
  for (i, (k, eps)) in enumerate(kAndEps)
    mapping[k] = i
  end

  return mapping
end

function bipartite_mapping(grid_size)
  @assert all(mod(dim, 2) == 0 for dim in grid_size)

  mapping = zeros(Int, grid_size...)

  blocks = Dict{Float64,Vector{NTuple{2,CartesianIndex}}}()
  offset = CartesianIndex([dim ÷ 2 for dim in grid_size]...)
  for (i, loc) in enumerate(CartesianIndices(mapping))
    partnerLoc = wrap_around(grid_size, loc + offset)

    if loc < partnerLoc
      continue
    end

    eps = epsilon(grid_size, loc)
    epsPartner = epsilon(grid_size, partnerLoc)

    if eps < epsPartner
      pair = loc, partnerLoc
    else
      pair = partnerLoc, loc
    end

    eps = max(eps, epsPartner)

    block = get!(blocks, eps, Vector{NTuple{2,CartesianIndex}}())
    if pair ∉ block
      push!(block, pair)
    end
  end

  i = 1
  for eps in sort([k for k in keys(blocks)])
    for pair in blocks[eps]
      mapping[pair[1]] = i
      mapping[pair[2]] = i + 1
      i += 2
    end
  end

  return mapping
end

# ## Constructing the MPO

using TimerOutputs
for grid_size in ((2, 2), (6, 6))
  let t = 1, U = 4, J = -0.5, mapping = bipartite_mapping(grid_size)
    @time "Constructing OpIDSum" sites, os = transcorrelated_fermi_hubbard(t, U, J, mapping)
    reset_timer!()
    @time "Constructing MPO" H = MPO_new(
      os, sites; combine_qn_sectors=true, output_level=0, check_for_errors=false
    )
    grid_size != (2, 2) && print_timer()
  end
end

# The transcorrelated momentum space Fermi-Hubbard MPO is very sparse. As such, giving Julia all the threads results in the best performance. The following timings are for the ``8 \times 8`` system on a computer with two Intel(R) Xeon(R) Gold 6438Y+ (64 total threads) and 250 GiB of memory.
#
# | Julia threads | BLAS Threads | OpIDSum time | MPO time |
# |---------------|--------------|--------------|----------|
# | 1             | 1            | 207s         | 2691s    |
# | 64            | 1            | 33s          | 1177s    |
#
# # Writing a Checkpoint File
#
# In certain cases (such as for the ``10 \times 10`` and ``12 \times 12`` systems), MPO construction can take so long that it is prudent to write out a checkpoint file in case of catastrophic failure. This can be accomplished by the `call_back` parameter to `MPO_new`, construction can be resumed later by calling `resume_MPO_construction`. This functionality is demonstrated below. 

using Serialization

function call_back(
  n::Int,
  H::MPO,
  sites::Vector{<:Index},
  llinks::Vector{<:Index},
  g::ITensorMPOConstruction.MPOGraph,
  op_cache_vec::OpCacheVec,
)::Nothing
  n != 18 && return nothing
  serialize("./mpo.jldump", (n, H, sites, llinks, g, op_cache_vec))
  println("Wrote a checkpoint to ./mpo.jldump")

  throw(InterruptException())

  return nothing
end

let t = 1, U = 4, J = -0.5, mapping = standard_mapping((6, 6))
  sites, os = transcorrelated_fermi_hubbard(t, U, J, mapping; conserve_momentum=false)
  try
    MPO_new(os, sites; combine_qn_sectors=true, check_for_errors=false, call_back)
  catch e
    if e isa InterruptException
      println("Caught a InterruptException!")
    else
      rethrow(e)
    end
  end

  println("Reading a checkpoint from ./mpo.jldump")
  n, H, sites, llinks, g, op_cache_vec = Serialization.deserialize("./mpo.jldump")
  H = resume_MPO_construction(
    Float64, n + 1, H, sites, llinks, g, op_cache_vec; combine_qn_sectors=true, call_back
  )
  println("Construction finished!")
  rm("./mpo.jldump")
end

# ````
# Wrote a checkpoint to ./mpo.jldump
# Caught a InterruptException!
# Reading a checkpoint from ./mpo.jldump
# Construction finished!
# ````
