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

Base.zero(::Type{OpID}) = OpID(0, 0)

Base.isless(op1::OpID, op2::OpID) = (op1.n, op1.id) < (op2.n, op2.id)

function sort_fermion_perm!(ops::AbstractVector{OpID}, op_cache_vec::OpCacheVec)::Int
  sign = +1
  for i in 2:length(ops)
    cur_op_is_fermionic = op_cache_vec[ops[i].n][ops[i].id].is_fermionic
    while i > 1 && ops[i].n < ops[i - 1].n
      if cur_op_is_fermionic && op_cache_vec[ops[i - 1].n][ops[i - 1].id].is_fermionic
        sign = -sign
      end

      ops[i - 1], ops[i] = ops[i], ops[i - 1]
      i -= 1
    end
  end

  return sign
end

mutable struct OpIDSum{N, C}
  _data::Vector{NTuple{N, OpID}}
  terms::Base.ReinterpretArray{OpID, 2, NTuple{N, OpID}, Vector{NTuple{N, OpID}}, true}
  scalars::Vector{C}
  num_terms::Threads.Atomic{Int}
  op_cache_vec::OpCacheVec
  abs_tol::C
  modify!::FunctionWrappers.FunctionWrapper{C, Tuple{SubArray{OpID, 1, Base.ReinterpretArray{OpID, 2, NTuple{N, OpID}, Vector{NTuple{N, OpID}}, true}, Tuple{UnitRange{Int64}, Int64}, false}}}
end

function OpIDSum{N, C}(max_terms::Int, op_cache_vec::OpCacheVec, f::Function; abs_tol::C=zero(C))::OpIDSum{N, C} where {N, C}
  data = Vector{NTuple{N, OpID}}(undef, max_terms)
  terms = reinterpret(reshape, OpID, data)
  f_wrapped = FunctionWrappers.FunctionWrapper{C, Tuple{SubArray{OpID, 1, Base.ReinterpretArray{OpID, 2, NTuple{N, OpID}, Vector{NTuple{N, OpID}}, true}, Tuple{UnitRange{Int64}, Int64}, false}}}(f)
  return OpIDSum(data, terms, zeros(C, max_terms), Threads.Atomic{Int}(0), op_cache_vec, abs_tol, f_wrapped)
end

function OpIDSum{N, C}(max_terms::Int, op_cache_vec::OpCacheVec; abs_tol::C=zero(C))::OpIDSum{N, C} where {N, C}
  return OpIDSum{N, C}(max_terms, op_cache_vec, ops -> 1; abs_tol)
end

function Base.length(os::OpIDSum)::Int
  return os.num_terms[]
end

function Base.eachindex(os::OpIDSum)::UnitRange{Int}
  return 1:length(os)
end

function Base.getindex(os::OpIDSum, i::Integer)
  return os.scalars[i], view(os.terms, :, i)
end

function ITensors.add!(os::OpIDSum{N, C}, scalar::C, ops)::OpIDSum{N, C} where {N, C}
  abs(scalar) <= os.abs_tol && return os

  num_terms = Threads.atomic_add!(os.num_terms, 1) + 1
  num_appended = 0
  for op in ops
    op.id == 1 && continue ## Filter out identity ops

    num_appended += 1
    os.terms[num_appended, num_terms] = op
  end

  for i in (num_appended + 1):N
    os.terms[i, num_terms] = zero(OpID)
  end

  permutation_sign = sort_fermion_perm!(view(os.terms, 1:num_appended, num_terms), os.op_cache_vec)

  scalar_modification = os.modify!(view(os.terms, 1:num_appended, num_terms))

  os.scalars[num_terms] = scalar * permutation_sign * scalar_modification

  return os
end

function ITensors.add!(os::OpIDSum{N, C}, scalar::C, ops::OpID...)::OpIDSum{N, C} where {N, C}
  return add!(os, scalar, ops)
end

function determine_val_type(os::OpIDSum{C}) where {C}
  !all(isreal(scalar) for scalar in os.scalars) && return ComplexF64
  !all(isreal(op.matrix) for ops_of_site in os.op_cache_vec for op in ops_of_site) &&
    return ComplexF64
  return Float64
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

@timeit function rewrite_in_operator_basis!(
  os::OpIDSum{N, C},
  basis_op_cache_vec::OpCacheVec
) where {N, C}
  
  op_cache_vec = os.op_cache_vec

  function scale_by_first_nz!(matrix::Matrix{ComplexF64})::ComplexF64
    for i in eachindex(matrix)
      entry = matrix[i]
      if entry != 0
        matrix ./= entry
        return entry
      end
    end
  end

  scaled_basis_ops = Vector{Tuple{ComplexF64, Matrix{ComplexF64}}}[Vector{Tuple{ComplexF64, Matrix{ComplexF64}}}() for _ in eachindex(basis_op_cache_vec)]
  for n in eachindex(basis_op_cache_vec)
    for op_info in basis_op_cache_vec[n]
      m = Matrix{ComplexF64}(op_info.matrix)
      scale = scale_by_first_nz!(m)
      push!(scaled_basis_ops[n], (scale, m))
    end
  end

  function convert_to_basis_memoized(
    ops::AbstractVector{OpID}
  )::Tuple{ComplexF64,Int}
    n = ops[1].n
    local_matrix = my_matrix(ops, op_cache_vec[n])
    scale = scale_by_first_nz!(local_matrix)

    for (i, (basis_scale, basis_matrix)) in enumerate(scaled_basis_ops[n])
      basis_matrix == local_matrix && return scale / basis_scale, i
    end

    return 0, 0
  end

  single_op_translation = Vector{Tuple{ComplexF64, Int}}[Vector{Tuple{ComplexF64, Int}}() for _ in eachindex(basis_op_cache_vec)]
  for n in eachindex(op_cache_vec)
    for id in eachindex(op_cache_vec[n])
      push!(single_op_translation[n], convert_to_basis_memoized([OpID(id, n)]))
    end
  end

  for i in eachindex(os)
    scalar, ops = os[i]

    for_equal_sites(ops) do a, b
      ops[a] == zero(OpID) && return

      if a == b
        coeff, basis_id = single_op_translation[ops[a].n][ops[a].id]
      else
        coeff, basis_id = convert_to_basis_memoized(view(ops, a:b))
      end

      if basis_id == 0
        error(
          "The following operator product cannot be simplified into a single basis operator.\n" *
          "\tOperator product = $(ops[a:b]), $ops\n",
        )
      end

      scalar *= coeff

      ops[a] = OpID(basis_id, ops[a].n)
      for k in (a + 1):b
        ops[k] = zero(OpID)
      end
    end

    os.scalars[i] = scalar
  end

  sort!(os.terms; dims=1, by=op -> op.n)
  os.op_cache_vec = basis_op_cache_vec

  return os
end

@timeit function op_sum_to_opID_sum(
  os::OpSum{C},
  sites::Vector{<:Index}
)::OpIDSum where {C}
  N = length(sites)

  ops_on_each_site = [Dict{Op,Int}(Op("I", n) => 1) for n in 1:N]
  op_cache_vec = [[OpInfo(Op("I", n), sites[n])] for n in 1:N]

  max_ops_per_term = maximum(length, ITensors.terms(os))
  opID_sum = OpIDSum{max_ops_per_term, C}(length(os), op_cache_vec)

  opID_term = Vector{OpID}()
  ## TODO: Don't need $i$ here
  for (i, term) in enumerate(os)
    resize!(opID_term, 0)
    for op in ITensors.terms(term)
      op.which_op == "I" && continue

      n = ITensors.site(op)

      if op ∉ keys(ops_on_each_site[n])
        ops_on_each_site[n][op] = length(ops_on_each_site[n]) + 1
        push!(op_cache_vec[n], OpInfo(op, sites[n]))
      end

      opID = ops_on_each_site[n][op]
      push!(opID_term, OpID(opID, n))
    end
  
    add!(opID_sum, ITensors.coefficient(term), opID_term)
  end

  return opID_sum
end

@timeit function check_for_errors(os::OpIDSum, op_cache_vec::OpCacheVec)::Nothing
  for i in eachindex(os)
    _, ops = os[i]

    flux = QN()
    fermion_parity = 0
    for j in eachindex(ops)
      opj = ops[j]
      opj == zero(OpID) && continue

      flux += op_cache_vec[opj.n][opj.id].qnFlux
      fermion_parity += op_cache_vec[opj.n][opj.id].is_fermionic

      if j < length(ops)
        ops[j + 1] == zero(OpID) && continue
        ops[j].n > ops[j + 1].n && error("The operators are not sorted by site: $ops")
        ops[j].n == ops[j + 1].n &&
          error("A site has more than one operator acting on it in a term: $ops")
      end
    end

    flux != QN() && error("The term does not have zero flux: $ops")
    mod(fermion_parity, 2) != 0 && error("Odd parity fermion terms not supported: $ops")
  end
end

@timeit function prepare_opID_sum!(
  os::OpIDSum,
  basis_op_cache_vec::Union{Nothing,OpCacheVec},
)
  if !isnothing(basis_op_cache_vec)
    rewrite_in_operator_basis!(os, basis_op_cache_vec)
  end

  # check_for_errors(os, op_cache_vec)
end

function my_matrix(term::AbstractVector{OpID}, op_cache::Vector{OpInfo})::Matrix{ComplexF64}
  @assert all(op.n == term[1].n for op in term)
  @assert !isempty(term)

  if length(term) == 1
    return copy(op_cache[term[1].id].matrix)
  end

  return prod(op_cache[op.id].matrix for op in term)
end
