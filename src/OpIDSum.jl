struct OpInfo
  matrix::Matrix
  isFermionic::Bool
  qnFlux::QN
end

function OpInfo(localOp::Op, site::Index)
  tensor = op(site, ITensors.which_op(localOp); ITensors.params(localOp)...)
  isFermionic = has_fermion_string(ITensors.name(localOp), site)
  qnFlux = flux(tensor)

  return OpInfo(copy(array(tensor)), isFermionic, isnothing(qnFlux) ? QN() : qnFlux)
end

OpCacheVec = Vector{Vector{OpInfo}}

function to_OpCacheVec(sites::Vector{<:Index}, ops::OpCacheVec)::OpCacheVec
  length(sites) != length(ops) &&
    error("Mismatch in the number of sites in `sites` and `ops`.")
  any(ops[i][1].matrix != I for i in 1:length(sites)) &&
    error("The first operator on each site must be the identity.")
  return ops
end

function to_OpCacheVec(sites::Vector{<:Index}, ops::Vector{Vector{String}})::OpCacheVec
  length(sites) != length(ops) &&
    error("Mismatch in the number of sites in `sites` and `ops`.")
  any(ops[i][1] != "I" for i in 1:length(sites)) &&
    error("The first operator on each site must be the identity.")
  return [[OpInfo(ITensors.Op(op, n), sites[n]) for op in ops[n]] for n in 1:length(sites)]
end

function to_OpCacheVec(sites::Vector{<:Index}, ::Nothing)::Nothing
  return nothing
end

struct OpID
  id::Int16
  n::Int16
end

struct OpIDSum{C}
  terms::Vector{OpID}
  offsets::Vector{Int}
  scalars::Vector{C}
end

function OpIDSum{C}()::OpIDSum{C} where {C}
  return OpIDSum(OpID[], Int[1], C[])
end

function Base.length(os::OpIDSum)::Int
  return length(os.scalars)
end

function Base.eachindex(os::OpIDSum)::UnitRange{Int}
  return 1:length(os)
end

function Base.getindex(os::OpIDSum, i::Integer)
  return os.scalars[i], @view os.terms[os.offsets[i]:(os.offsets[i + 1] - 1)]
end

function Base.push!(os::OpIDSum{C}, scalar::C, ops)::OpIDSum{C} where {C}
  append!(os.terms, ops)
  push!(os.offsets, os.offsets[end] + length(ops))
  push!(os.scalars, scalar)

  return os
end

function Base.push!(os::OpIDSum{C}, scalar::C, ops::OpID...)::OpIDSum{C} where {C}
  return push!(os, scalar, ops)
end

function Base.push!(os::OpIDSum{C}, scalar::C, op::OpID)::OpIDSum{C} where {C}
  push!(os.terms, op)
  push!(os.offsets, os.offsets[end] + 1)
  push!(os.scalars, scalar)

  return os
end

function set_scalar!(os::OpIDSum{C}, i::Integer, scalar::C)::Nothing where {C}
  os.scalars[i] = scalar
  return nothing
end

function add_to_scalar!(os::OpIDSum{C}, i::Integer, scalar::C)::Nothing where {C}
  os.scalars[i] += scalar
  return nothing
end

function determine_val_type(os::OpIDSum{C}, opCacheVec::OpCacheVec) where {C}
  !all(isreal(scalar) for scalar in os.scalars) && return ComplexF64
  !all(isreal(op.matrix) for opsOfSite in opCacheVec for op in opsOfSite) &&
    return ComplexF64
  return Float64
end

@timeit function sort_each_term!(
  os::OpIDSum, opCacheVec::OpCacheVec, sites::Vector{<:Index}
)::Nothing
  isless_site(o1::OpID, o2::OpID) = o1.n < o2.n

  perm = Vector{Int}()
  for j in eachindex(os)
    scalar, ops = os[j]
    Nt = length(ops)

    # Sort operators by site order,
    # and keep the permutation used, perm, for analysis below
    resize!(perm, Nt)
    sortperm!(perm, ops; alg=InsertionSort, lt=isless_site)
    ops .= ops[perm]

    # Identify fermionic operators,
    # zeroing perm for bosonic operators,
    parity = +1
    for (i, op) in enumerate(ops)
      op.n > length(sites) && error(
        "The OpSum contains an operator acting on site $(op.n) that extends beyond the number of sites $(length(sites)).",
      )

      if opCacheVec[op.n][op.id].isFermionic
        parity = -parity
      else
        # Ignore bosonic operators in perm
        # by zeroing corresponding entries
        perm[i] = 0
      end
    end

    if parity == -1
      error("Parity-odd fermionic terms not yet supported by AutoMPO")
    end

    # Keep only fermionic op positions (non-zero entries)
    filter!(!iszero, perm)

    # and account for anti-commuting, fermionic operators
    # during above sort; put resulting sign into coef
    set_scalar!(os, j, ITensors.parity_sign(perm) * scalar)
  end
end

@timeit function merge_terms!(os::OpIDSum{C})::Nothing where {C}
  uniqueTermLocations = Dict{
    SubArray{OpID,1,Vector{OpID},Tuple{UnitRange{Int64}},true},Int
  }()
  for i in eachindex(os)
    scalar, ops = os[i]

    loc = get!(uniqueTermLocations, ops, i)
    loc == i && continue

    add_to_scalar!(os, loc, scalar)
    set_scalar!(os, i, zero(C))
  end

  return nothing
end

function convert_to_basis(op::Matrix, basisCache::Vector{OpInfo})::Tuple{ComplexF64,Int}
  function check_ratios_are_equal(a1::Number, b1::Number, ai::Number, bi::Number)::Bool
    # TODO: Add a tolerance param here
    return abs(a1 * bi - b1 * ai) <= 1e-10
  end

  for i in eachindex(basisCache)
    basisOp = basisCache[i].matrix

    j = findfirst(val -> val != 0, basisOp)

    if all(
      check_ratios_are_equal(basisOp[j], op[j], basisOp[k], op[k]) for
      k in CartesianIndices(op)
    )
      return op[j] / basisOp[j], i
    end
  end

  return 0, 0
end

function for_equal_sites(f::Function, ops::AbstractVector{OpID})::Nothing
  i = 1
  while i <= length(ops)
    j = i
    while j <= length(ops)
      (j == length(ops) || ops[i].n != ops[j + 1].n) && break
      j += 1
    end

    f(i, j)
    i = j + 1
  end

  return nothing
end

@timeit function rewrite_in_operator_basis(
  os::OpIDSum{C}, opCacheVec::OpCacheVec, basisOpCacheVec::OpCacheVec
)::OpIDSum{C} where {C}
  @memoize Dict function convert_to_basis_memoized(
    ops::AbstractVector{OpID}
  )::Tuple{ComplexF64,Int}
    n = ops[1].n
    localOp = my_matrix(ops, opCacheVec[n])
    return convert_to_basis(localOp, basisOpCacheVec[n])
  end

  newOpIdSum = OpIDSum{C}()

  opsInBasis = Vector{OpID}()
  for i in eachindex(os)
    scalar, ops = os[i]

    empty!(opsInBasis)

    for_equal_sites(ops) do a, b
      coeff, basisID = convert_to_basis_memoized(@view ops[a:b])

      if basisID == 0
        error(
          "The following operator product cannot be simplified into a single basis operator.\n" *
          "\tOperator product = $(ops[a:b])\n",
        )
      end

      scalar *= coeff
      push!(opsInBasis, OpID(basisID, ops[a].n))
    end

    push!(newOpIdSum, C(scalar), opsInBasis)
  end

  return newOpIdSum
end

@timeit function op_sum_to_opID_sum(
  os::OpSum{C}, sites::Vector{<:Index}
)::Tuple{OpIDSum{C},OpCacheVec} where {C}
  N = length(sites)

  opsOnEachSite = [Dict{Op,Int}(Op("I", n) => 1) for n in 1:N]

  opID_sum = OpIDSum{C}()

  opIDTerm = Vector{OpID}()
  for (i, term) in enumerate(os)
    resize!(opIDTerm, 0)
    for op in ITensors.terms(term)
      op.which_op == "I" && continue

      n = ITensors.site(op)
      opID = get!(opsOnEachSite[n], op, length(opsOnEachSite[n]) + 1)
      push!(opIDTerm, OpID(opID, n))
    end

    push!(opID_sum, ITensors.coefficient(term), opIDTerm)
  end

  opCacheVec = OpCacheVec()
  for (n, opsOnSite) in enumerate(opsOnEachSite)
    opCache = Vector{OpInfo}(undef, length(opsOnSite))

    for (op, id) in opsOnSite
      @assert ITensors.site(op) == n
      opCache[id] = OpInfo(op, sites[n])
    end

    push!(opCacheVec, opCache)
  end

  return opID_sum, opCacheVec
end

@timeit function check_for_errors(os::OpIDSum, opCacheVec::OpCacheVec)::Nothing
  for i in eachindex(os)
    _, ops = os[i]

    flux = QN()
    fermionParity = 0
    for j in eachindex(ops)
      opj = ops[j]
      flux += opCacheVec[opj.n][opj.id].qnFlux
      fermionParity += opCacheVec[opj.n][opj.id].isFermionic

      if j < length(ops)
        ops[j].n > ops[j + 1].n && error("The operators are not sorted by site.")
        ops[j].n == ops[j + 1].n &&
          error("A site has more than one operator acting on it in a term.")
      end
    end

    flux != QN() && error("The term does not have zero flux.")
    mod(fermionParity, 2) != 0 && error("Odd parity fermion terms not supported.")
  end
end

@timeit function prepare_opID_sum!(
  os::OpIDSum,
  sites::Vector{<:Index},
  opCacheVec::OpCacheVec,
  basisOpCacheVec::Union{Nothing,OpCacheVec},
)::Tuple{OpIDSum,OpCacheVec}
  sort_each_term!(os, opCacheVec, sites)
  merge_terms!(os)

  if !isnothing(basisOpCacheVec)
    os, opCacheVec = rewrite_in_operator_basis(os, opCacheVec, basisOpCacheVec),
    basisOpCacheVec
    merge_terms!(os)
  end

  check_for_errors(os, opCacheVec)

  # This is to account for the dummy site at the end
  push!(opCacheVec, OpInfo[])

  return os, opCacheVec
end

function my_matrix(
  term::AbstractVector{OpID}, opCache::Vector{OpInfo}; needsJWString::Bool=false
)
  @assert all(op.n == term[1].n for op in term)

  if isempty(term)
    localMatrix = Matrix{Float64}(I, size(opCache[begin].matrix)...)
  else
    localMatrix = prod(opCache[op.id].matrix for op in term)
  end

  if needsJWString
    # localMatrix can be returned directly from the opCache, and in this case we need to copy it
    # before modification.
    localMatrix = copy(localMatrix)

    # This is a weird way of applying the Jordan-Wigner string it but op("F", sites[n]) is very slow.
    if size(localMatrix, 1) == 2
      localMatrix[:, 2] *= -1
    elseif size(localMatrix, 1) == 4
      localMatrix[:, 2] *= -1
      localMatrix[:, 3] *= -1
    else
      error("Unknown fermionic site.")
    end
  end

  return localMatrix
end
