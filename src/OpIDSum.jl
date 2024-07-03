struct OpInfo
  matrix::Matrix
  is_fermionic::Bool
  qnFlux::QN
end

function OpInfo(local_op::Op, site::Index)
  tensor = op(site, ITensors.which_op(local_op); ITensors.params(local_op)...)
  is_fermionic = has_fermion_string(ITensors.name(local_op), site)
  qnFlux = flux(tensor)

  return OpInfo(copy(array(tensor)), is_fermionic, isnothing(qnFlux) ? QN() : qnFlux)
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

Base.isless(op1::OpID, op2::OpID) = (op1.n, op1.id) < (op2.n, op2.id)

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
  num_appended = 0
  for op in ops
    op.id == 1 && continue ## Filter out identity ops
    push!(os.terms, op)
    num_appended += 1
  end

  push!(os.offsets, os.offsets[end] + num_appended)
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

function determine_val_type(os::OpIDSum{C}, op_cache_vec::OpCacheVec) where {C}
  !all(isreal(scalar) for scalar in os.scalars) && return ComplexF64
  !all(isreal(op.matrix) for ops_of_site in op_cache_vec for op in ops_of_site) &&
    return ComplexF64
  return Float64
end

@timeit function sort_each_term!(
  os::OpIDSum, op_cache_vec::OpCacheVec, sites::Vector{<:Index}
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

      if op_cache_vec[op.n][op.id].is_fermionic
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
  unique_location = Dict{SubArray{OpID,1,Vector{OpID},Tuple{UnitRange{Int64}},true},Int}()
  for i in eachindex(os)
    scalar, ops = os[i]

    loc = get!(unique_location, ops, i)
    loc == i && continue

    add_to_scalar!(os, loc, scalar)
    set_scalar!(os, i, zero(C))
  end

  return nothing
end

function convert_to_basis(op::Matrix, basis_cache::Vector{OpInfo})::Tuple{ComplexF64,Int}
  function check_ratios_are_equal(a1::Number, b1::Number, ai::Number, bi::Number)::Bool
    # TODO: Add a tolerance param here
    return abs(a1 * bi - b1 * ai) <= 1e-10
  end

  for i in eachindex(basis_cache)
    basis_op = basis_cache[i].matrix

    j = findfirst(val -> val != 0, basis_op)

    if all(
      check_ratios_are_equal(basis_op[j], op[j], basis_op[k], op[k]) for
      k in CartesianIndices(op)
    )
      return op[j] / basis_op[j], i
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
  os::OpIDSum{C}, op_cache_vec::OpCacheVec, basis_op_cache_vec::OpCacheVec
)::OpIDSum{C} where {C}
  @memoize Dict function convert_to_basis_memoized(
    ops::AbstractVector{OpID}
  )::Tuple{ComplexF64,Int}
    n = ops[1].n
    local_op = my_matrix(ops, op_cache_vec[n])
    return convert_to_basis(local_op, basis_op_cache_vec[n])
  end

  new_os = OpIDSum{C}()

  ops_in_basis = Vector{OpID}()
  for i in eachindex(os)
    scalar, ops = os[i]

    empty!(ops_in_basis)

    for_equal_sites(ops) do a, b
      coeff, basis_id = convert_to_basis_memoized(@view ops[a:b])

      if basis_id == 0
        error(
          "The following operator product cannot be simplified into a single basis operator.\n" *
          "\tOperator product = $(ops[a:b])\n",
        )
      end

      scalar *= coeff
      push!(ops_in_basis, OpID(basis_id, ops[a].n))
    end

    push!(new_os, C(scalar), ops_in_basis)
  end

  return new_os
end

@timeit function op_sum_to_opID_sum(
  os::OpSum{C}, sites::Vector{<:Index}
)::Tuple{OpIDSum{C},OpCacheVec} where {C}
  N = length(sites)

  ops_on_each_site = [Dict{Op,Int}(Op("I", n) => 1) for n in 1:N]

  opID_sum = OpIDSum{C}()

  opID_term = Vector{OpID}()
  ## TODO: Don't need $i$ here
  for (i, term) in enumerate(os)
    resize!(opID_term, 0)
    for op in ITensors.terms(term)
      op.which_op == "I" && continue

      n = ITensors.site(op)
      opID = get!(ops_on_each_site[n], op, length(ops_on_each_site[n]) + 1)
      push!(opID_term, OpID(opID, n))
    end

    push!(opID_sum, ITensors.coefficient(term), opID_term)
  end

  op_cache_vec = OpCacheVec()
  for (n, ops_on_site) in enumerate(ops_on_each_site)
    op_cache = Vector{OpInfo}(undef, length(ops_on_site))

    for (op, id) in ops_on_site
      @assert ITensors.site(op) == n
      op_cache[id] = OpInfo(op, sites[n])
    end

    push!(op_cache_vec, op_cache)
  end

  return opID_sum, op_cache_vec
end

@timeit function check_for_errors(os::OpIDSum, op_cache_vec::OpCacheVec)::Nothing
  for i in eachindex(os)
    _, ops = os[i]

    flux = QN()
    fermion_parity = 0
    for j in eachindex(ops)
      opj = ops[j]
      flux += op_cache_vec[opj.n][opj.id].qnFlux
      fermion_parity += op_cache_vec[opj.n][opj.id].is_fermionic

      if j < length(ops)
        ops[j].n > ops[j + 1].n && error("The operators are not sorted by site.")
        ops[j].n == ops[j + 1].n &&
          error("A site has more than one operator acting on it in a term.")
      end
    end

    flux != QN() && error("The term does not have zero flux.")
    mod(fermion_parity, 2) != 0 && error("Odd parity fermion terms not supported.")
  end
end

@timeit function prepare_opID_sum!(
  os::OpIDSum,
  sites::Vector{<:Index},
  op_cache_vec::OpCacheVec,
  basis_op_cache_vec::Union{Nothing,OpCacheVec},
)::Tuple{OpIDSum,OpCacheVec}
  sort_each_term!(os, op_cache_vec, sites)
  merge_terms!(os)

  if !isnothing(basis_op_cache_vec)
    os, op_cache_vec = rewrite_in_operator_basis(os, op_cache_vec, basis_op_cache_vec),
    basis_op_cache_vec
    merge_terms!(os)
  end

  check_for_errors(os, op_cache_vec)

  # This is to account for the dummy site at the end
  push!(op_cache_vec, OpInfo[])

  return os, op_cache_vec
end

function my_matrix(
  term::AbstractVector{OpID}, op_cache::Vector{OpInfo}; needs_JW_string::Bool=false
)
  @assert all(op.n == term[1].n for op in term)

  if isempty(term)
    local_matrix = Matrix{Float64}(I, size(op_cache[begin].matrix)...)
  else
    local_matrix = prod(op_cache[op.id].matrix for op in term)
  end

  if needs_JW_string
    # local_matrix can be returned directly from the op_cache, and in this case we need to copy it
    # before modification.
    local_matrix = copy(local_matrix)

    # This is a weird way of applying the Jordan-Wigner string it but op("F", sites[n]) is very slow.
    if size(local_matrix, 1) == 2
      local_matrix[:, 2] *= -1
    elseif size(local_matrix, 1) == 4
      local_matrix[:, 2] *= -1
      local_matrix[:, 3] *= -1
    else
      error("Unknown fermionic site.")
    end
  end

  return local_matrix
end
