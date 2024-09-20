BlockSparseMatrix{C} = Dict{Tuple{Int,Int},Matrix{C}}

MPOGraph{N, C, Ti} = BipartiteGraph{LeftVertex, NTuple{N, OpID{Ti}}, C}

## Taken from https://discourse.julialang.org/t/how-to-sort-two-or-more-lists-at-once/12073/13
struct CoSorterElement{T1,T2}
  x::T1
  y::T2
end

struct CoSorter{T1,T2,S<:AbstractArray{T1},C<:AbstractArray{T2}} <: AbstractVector{CoSorterElement{T1,T2}}
  sortarray::S
  coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)

Base.getindex(c::CoSorter, i...) = CoSorterElement(getindex(c.sortarray, i...), getindex(c.coarray, i...))

Base.setindex!(c::CoSorter, t::CoSorterElement, i...) = (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c)

Base.isless(a::CoSorterElement, b::CoSorterElement) = isless(a.x, b.x)


@timeit function MPOGraph(os::OpIDSum{N, C, Ti})::MPOGraph{N, C, Ti} where {N, C, Ti}
  for i in 1:length(os)
    for j in size(os.terms, 1):-1:1
      if os.terms[j, i] != zero(os.terms[j, i])
        reverse!(view(os.terms, 1:j, i))
        break
      end
    end
  end
  
  @timeit "sorting" let
    resize!(os._data, length(os))
    resize!(os.scalars, length(os))
    c = CoSorter(os._data, os.scalars)
    sort!(c; alg=QuickSort)
  end

  for i in 1:(length(os) - 1)
    if os._data[i] == os._data[i + 1]
      os.scalars[i + 1] += os.scalars[i]
      os.scalars[i] = 0
    end
  end

  nnz = 0
  for i in eachindex(os)
    if abs(os.scalars[i]) > os.abs_tol
      nnz += 1
      os.scalars[nnz] = os.scalars[i]
      os._data[nnz] = os._data[i]
    end
  end

  os.num_terms[] = nnz
  resize!(os._data, nnz)
  resize!(os.scalars, nnz)

  g = MPOGraph{N, C, Ti}([], os._data, [])

  ## TODO: Break this out into a function that is shared with at_site!
  next_edges = [Vector{Tuple{Int, C}}() for _ in 1:length(os.op_cache_vec[1])]

  for j in 1:right_size(g)
    weight = os.scalars[j]
    onsite_op = get_onsite_op(right_vertex(g, j), 1)
    push!(next_edges[onsite_op], (j, weight))
  end

  for op_id in 1:length(os.op_cache_vec[1])
    isempty(next_edges[op_id]) && continue

    first_rv_id = next_edges[op_id][1][1]
    needs_JW_string = is_fermionic(right_vertex(g, first_rv_id), 2, os.op_cache_vec)
    push!(g.left_vertices, LeftVertex(1, op_id, needs_JW_string))
    push!(g.edges_from_left, next_edges[op_id])
  end

  return g
end

function get_column!(Q::SparseArrays.SPQR.QRSparseQ, col::Int, res::Vector)
  res .= 0
  res[col] = 1

  for l in size(Q.factors, 2):-1:1
    τl = -Q.τ[l]
    h = view(Q.factors, :, l)
    axpy!(τl*dot(h, res), h, res)
  end

  return res
end


function sparse_qr(
  A::SparseMatrixCSC, tol::Real
)::Tuple{SparseArrays.SPQR.QRSparseQ,SparseMatrixCSC,Vector{Int},Vector{Int},Int}
  ret = qr(A; tol=tol)
  return ret.Q, ret.R, ret.prow, ret.pcol, rank(ret)
end

function for_non_zeros_batch(f::Function, A::SparseMatrixCSC, max_col::Int)::Nothing
  @assert max_col <= size(A, 2) "$max_col, $(size(A, 2))"

  rows = rowvals(A)
  vals = nonzeros(A)
  for col in 1:max_col
    range = nzrange(A, col)
    isempty(range) && continue
    f((@view vals[range]), (@view rows[range]), col)
  end
end

function for_non_zeros_batch(f::Function, Q::SparseArrays.SPQR.QRSparseQ, max_col::Int)::Nothing
  @assert max_col <= size(Q, 2) "$max_col, $(size(Q, 2))"

  res = zeros(eltype(Q), size(Q, 1))
  for col in 1:max_col
    get_column!(Q, col, res)
    f(res, col)
  end
end

function column_one_norms(A::SparseArrays.SPQR.QRSparseQ, max_col::Int)::Vector{Float64}
  sums = zeros(max_col)
  for_non_zeros_batch(A, max_col) do entries, col
    sums[col] = sum(abs, entries)
  end

  return sums
end

function row_one_norms(A::SparseMatrixCSC, max_row::Int)::Vector{Float64}
  sums = zeros(max_row)

  for_non_zeros_batch(A, size(A, 2)) do entries, rows, col
    @assert issorted(rows)
    for (entry, row) in zip(entries, rows)
      row > max_row && return
      sums[row] += abs(entry)
    end
  end

  return sums
end

function add_to_local_matrix!(a::Matrix, weight::Number, local_op::Matrix, needs_JW_string::Bool)::Nothing
  if !needs_JW_string
    a .+= weight * local_op
  elseif size(local_op, 1) == 2
    a[:, 1] .+= weight * local_op[:, 1]
    a[:, 2] .-= weight * local_op[:, 2]
  elseif size(local_op, 1) == 4
    a[:, 1] .+= weight * local_op[:, 1]
    a[:, 2] .-= weight * local_op[:, 2]
    a[:, 3] .-= weight * local_op[:, 3]
    a[:, 4] .+= weight * local_op[:, 4]
  else
    error("Unknown fermionic site.")
  end

  return nothing
end

function merge_qn_sectors(qi_of_cc::Vector{Pair{QN, Int}})::Tuple{Vector{Int}, Vector{Pair{QN, Int}}}
  new_order = sortperm(qi_of_cc, by = pair -> pair[1])
  qi_of_cc = sort(qi_of_cc, by = pair -> pair[1])
  
  new_qi = Pair{QN, Int}[qi_of_cc[1]]
  for qi in view(qi_of_cc, 2:length(qi_of_cc))
    if qi.first == new_qi[end].first
      new_qi[end] = qi.first => new_qi[end].second + qi.second
    else
      push!(new_qi, qi)
    end
  end

  return new_order, new_qi
end

function find_first_eq_rv(g::MPOGraph, j::Int, n::Int)::Int
  while j > 1 && are_equal(right_vertex(g, j), right_vertex(g, j - 1), n)
    j -= 1
  end

  return j
end

@timeit function at_site!(
  ValType::Type{<:Number},
  g::MPOGraph{N, C, Ti},
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  left_norm2s::Vector{Float64},
  op_cache_vec::OpCacheVec;
  combine_qn_sectors::Bool=false,
  redistribute_weight::Bool=false,
  output_level::Int=0)::Tuple{MPOGraph{N, C, Ti},Vector{Int},Vector{BlockSparseMatrix{ValType}},Index} where {N, C, Ti}

  has_qns = hasqns(sites)
  
  # combine_duplicate_adjacent_right_vertices!(g, terms_eq_from(n + 1))
  ccs = compute_connected_components(g)
  nccs = num_connected_components(ccs)

  rank_of_cc = zeros(Int, nccs)
  matrix_of_cc = [BlockSparseMatrix{ValType}() for _ in 1:nccs]
  qi_of_cc = Pair{QN,Int}[QN() => 0 for _ in 1:nccs]
  next_edges_of_cc = [Matrix{Vector{Tuple{Int, C}}}(undef, 0, 0) for _ in 1:nccs]
  next_left_norm2s_of_cc = [Vector{Float64}() for _ in 1:nccs]

  tol == -1 && (tol = get_default_tol(g, ccs))

  output_level > 0 && println("  The graph is $(left_size(g)) × $(right_size(g)) with $(num_edges(g)) edges and $(nccs) connected components. tol = $(@sprintf("%.2E", tol))")

  @timeit "Threaded loop" Threads.@threads for cc in 1:nccs
    if left_size(ccs, cc) == 1
      lv_id = only(ccs.lvs_of_component[cc])

      left_map = ccs.lvs_of_component[cc]
      Q = qr(sparse(reshape([one(C)], 1, 1)); tol=0).Q
      prow = [1]
      rank = 1

      lambdas = Float64[1.0]

      first_rv_id, _ = g.edges_from_left[lv_id][1]
    else
      W, left_map, right_map = get_cc_matrix(g, ccs, cc; clear_edges=true)

      ## Compute the decomposition and then free W
      Q, R, prow, pcol, rank = sparse_qr(W, tol, left_norms_of_these_specific_entries)
      W = nothing

      ## TODO: Not even sure if this is a bottleneck, but it could be done better
      if redistribute_weight
        scaled_Q_col_norms = column_one_norms(Q, rank) ./ size(Q, 1)
        scaled_R_row_norms = row_one_norms(R, rank) ./ size(R, 2)
        lambdas = [sqrt(scaled_R_row_norms[m] / scaled_Q_col_norms[m]) for m in 1:rank]
      else
        lambdas = ones(rank)
      end

      first_rv_id = right_map[1]
    end

    rank_of_cc[cc] = rank

    if has_qns
      right_flux = flux(right_vertex(g, first_rv_id), n + 1, op_cache_vec)
      qi_of_cc[cc] = (QN() - right_flux) => rank
    end

    next_left_norm2s_of_cc[cc] = Vector{Float64}(undef, rank)

    # Form the local transformation tensor.
    for_non_zeros_batch(Q, rank) do weights, m
      cur_norm2 = 0
      for (i, weight) in enumerate(weights)
        weight == 0 && continue

        lv = left_vertex(g, left_map[prow[i]])
        local_op = op_cache_vec[n][lv.op_id].matrix

        matrix_element = get!(matrix_of_cc[cc], (lv.link, m)) do
          return zeros(C, dim(sites[n]), dim(sites[n]))
        end

        cur_norm2 += abs2(weight) * left_norm2s[lv.link]

        weight *= lambdas[m]
        add_to_local_matrix!(matrix_element, weight, local_op, lv.needs_JW_string)
      end

      next_left_norm2s_of_cc[cc][m] = lambdas[m]^2 * cur_norm2
    end

    Q = nothing
    prow = nothing

    # If we are at the last site, then R will be a 1x1 matrix containing an overall scaling.
    # We can also skip building the next graph.
    if n == length(sites)
      if left_size(ccs, cc) == 1
        scaling = only(g.edges_from_left[lv_id])[2]
      else
        scaling = only(R)
      end

      for block in values(matrix_of_cc[cc])
        block .*= scaling / lambdas[1]
      end

      continue
    end

    # Build the graph for the next site.
    next_edges = Matrix{Vector{Tuple{Int, C}}}(undef, rank, length(op_cache_vec[n + 1]))
    for i in eachindex(next_edges)
      next_edges[i] = Vector{Tuple{Int, C}}()
    end

    if left_size(ccs, cc) == 1
      for (j, weight) in g.edges_from_left[lv_id]
        weight /= lambdas[1]

        op_id = get_onsite_op(right_vertex(g, j), n + 1)

        rv_id = find_first_eq_rv(g, j, n + 2)

        push!(next_edges[1, op_id], (rv_id, weight))
      end

      empty!(g.edges_from_left[lv_id])
      sizehint!(g.edges_from_left[lv_id], 0)
    else
      for_non_zeros_batch(R, length(right_map)) do weights, ms, j
        j = right_map[pcol[j]]
        op_id = get_onsite_op(right_vertex(g, j), n + 1)
        
        rv_id = find_first_eq_rv(g, j, n + 2)

        @assert issorted(ms)
        for (weight, m) in zip(weights, ms)
          m > rank && return

          weight /= lambdas[m]
          push!(next_edges[m, op_id], (rv_id, weight))
        end
      end
    end

    next_edges_of_cc[cc] = next_edges
  end

  cc_order = [i for i in 1:nccs]
  if combine_qn_sectors && has_qns
    cc_order, qi_of_cc = merge_qn_sectors(qi_of_cc)
  end

  next_graph = MPOGraph{N, C, Ti}([], g.right_vertices, [])
  offset_of_cc = zeros(Int, nccs + 1)

  cur_offset = 0
  for (i, cc) in enumerate(cc_order)
    offset_of_cc[cc] = cur_offset

    next_edges = next_edges_of_cc[cc]
    for op_id in 1:size(next_edges, 2)
      for m in 1:size(next_edges, 1)
        isempty(next_edges[m, op_id]) && continue

        first_rv_id = next_edges[m, op_id][1][1]
        needs_JW_string = is_fermionic(right_vertex(g, first_rv_id), n + 2, op_cache_vec)
        push!(next_graph.left_vertices, LeftVertex(m + cur_offset, op_id, needs_JW_string))
        push!(next_graph.edges_from_left, next_edges[m, op_id])
      end
    end

    cur_offset += rank_of_cc[cc]
  end

  if has_qns
    outgoing_link = Index(qi_of_cc...; tags="Link,l=$n", dir=ITensors.Out)
    output_level > 1 && println("    Total rank is $cur_offset with $(length(qi_of_cc)) different QN sectors.")
  else
    outgoing_link = Index(cur_offset; tags="Link,l=$n")
    output_level > 1 && println("    Total rank is $cur_offset.")
  end

  return next_graph, offset_of_cc, matrix_of_cc, outgoing_link
end
