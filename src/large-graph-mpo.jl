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

function find_first_eq_rv(g::MPOGraph, j::Int, n::Int)::Int
  while j > 1 && are_equal(right_vertex(g, j), right_vertex(g, j - 1), n)
    j -= 1
  end

  return j
end

function build_next_edges_specialization!(
  next_edges::Matrix{Vector{Tuple{Int, C}}},
  g::MPOGraph{N, C, Ti},
  cur_site::Int,
  edges
)::Nothing where {N, C, Ti}
  @assert size(next_edges, 1) == 1

  for (rv_id, weight) in edges
    op_id = get_onsite_op(right_vertex(g, rv_id), cur_site + 1)

    rv_id = find_first_eq_rv(g, rv_id, cur_site + 2)

    push!(next_edges[1, op_id], (rv_id, weight))
  end

  return nothing
end

function add_to_next_graph!(
  next_graph::MPOGraph{N, C, Ti},
  cur_graph::MPOGraph{N, C, Ti},
  op_cache_vec::OpCacheVec,
  cur_site::Int,
  cur_offset::Int,
  next_edges::Matrix{Vector{Tuple{Int, C}}}
)::Nothing where {N, C, Ti}
  for op_id in 1:size(next_edges, 2)
    for m in 1:size(next_edges, 1)
      cur_edges = next_edges[m, op_id]
      isempty(cur_edges) && continue

      first_rv_id = cur_edges[1][1]
      needs_JW_string = is_fermionic(right_vertex(cur_graph, first_rv_id), cur_site + 2, op_cache_vec)
      push!(next_graph.left_vertices, LeftVertex(m + cur_offset, op_id, needs_JW_string))
      push!(next_graph.edges_from_left, cur_edges)
    end
  end

  return nothing
end

@timeit function MPOGraph(os::OpIDSum{N, C, Ti})::MPOGraph{N, C, Ti} where {N, C, Ti}
  ## Reverse the terms in the sum, ignoring trailing identity operators.
  for i in 1:length(os)
    @assert size(os.terms, 1) == N
    for j in N:-1:1
      if os.terms[j, i] != zero(os.terms[j, i])
        reverse!(view(os.terms, 1:j, i))
        break
      end
    end
  end
  
  ## Sort the terms and scalars.
  @timeit "sorting" let
    resize!(os._data, length(os))
    resize!(os.scalars, length(os))
    c = CoSorter(os._data, os.scalars)
    sort!(c; alg=QuickSort)
  end

  ## Combine duplicate terms.
  for i in 1:(length(os) - 1)
    if os._data[i] == os._data[i + 1]
      os.scalars[i + 1] += os.scalars[i]
      os.scalars[i] = 0
    end
  end

  ## Remove terms which are below the tolerance.
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

  next_edges = Matrix{Vector{Tuple{Int, C}}}(undef, 1, length(os.op_cache_vec[1]))
  for i in eachindex(next_edges)
    next_edges[i] = Vector{Tuple{Int, C}}()
  end


  build_next_edges_specialization!(next_edges, g, 0, enumerate(os.scalars))

  add_to_next_graph!(g, g, os.op_cache_vec, 0, 0, next_edges)

  return g
end

function sparse_qr(
  A::SparseMatrixCSC, tol::Real, absoluteTol::Bool
)::Tuple{SparseArrays.SPQR.QRSparseQ,SparseMatrixCSC,Vector{Int},Vector{Int},Int}
  ret = nothing

  if !absoluteTol
    tol *= SparseArrays.SPQR._default_tol(A)
  end

  SparseArrays.CHOLMOD.@cholmod_param SPQR_nthreads = 1 begin
    ret = qr(A; tol)
  end

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

  res = zeros(eltype(Q), size(Q, 1))
  for col in 1:max_col
    get_column!(Q, col, res)
    f(res, col)
  end
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

@timeit function at_site!(
  ValType::Type{<:Number},
  g::MPOGraph{N, C, Ti},
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  absoluteTol::Bool,
  op_cache_vec::OpCacheVec;
  combine_qn_sectors::Bool=false,
  output_level::Int=0)::Tuple{MPOGraph{N, C, Ti},Vector{Int},Vector{BlockSparseMatrix{ValType}},Index} where {N, C, Ti}

  has_qns = hasqns(sites)
  
  ccs = compute_connected_components(g)
  nccs = num_connected_components(ccs)

  ## The rank of each connected component.
  rank_of_cc = zeros(Int, nccs)

  ## The MPO tensor for each component.
  matrix_of_cc = [BlockSparseMatrix{ValType}() for _ in 1:nccs] 

  ## The QN of each component
  qi_of_cc = Pair{QN,Int}[QN() => 0 for _ in 1:nccs]

  ## A map from the incoming link to the next site (outgoing link from this site) and the
  ## operator on the next site (this uniquely specifies the left vertex of the next site)
  ## to the right vertices it will connect to along with the weight.
  next_edges_of_cc = [Matrix{Vector{Tuple{Int, C}}}(undef, 0, 0) for _ in 1:nccs]

  output_level > 0 && println("  The graph is $(left_size(g)) × $(right_size(g)) with $(num_edges(g)) edges and $(nccs) connected components. tol = $(@sprintf("%.2E", tol))")

  @timeit "Threaded loop" Threads.@threads for cc in 1:nccs
    ## A specialization for when there is only one vertex on the left. This is
    ## a very common case that can be sped up significantly.
    if left_size(ccs, cc) == 1
      lv_id = only(ccs.lvs_of_component[cc])

      left_map = ccs.lvs_of_component[cc]
      Q = qr(sparse(reshape([one(C)], 1, 1)); tol=0).Q
      prow = [1]
      rank = 1

      first_rv_id, _ = g.edges_from_left[lv_id][1]
    else
      W, left_map, right_map = get_cc_matrix(g, ccs, cc; clear_edges=true)

      ## Compute the decomposition and then free W
      Q, R, prow, pcol, rank = sparse_qr(W, tol, absoluteTol)
      W = nothing

      first_rv_id = right_map[1]
    end

    rank_of_cc[cc] = rank

    ## Compute and store the QN of this component
    if has_qns
      right_flux = flux(right_vertex(g, first_rv_id), n + 1, op_cache_vec)
      qi_of_cc[cc] = (QN() - right_flux) => rank
    end

    ## Form the local transformation tensor.
    for_non_zeros_batch(Q, rank) do weights, m
      for (i, weight) in enumerate(weights)
        weight == 0 && continue

        lv = left_vertex(g, left_map[prow[i]])
        local_op = op_cache_vec[n][lv.op_id].matrix

        matrix_element = get!(matrix_of_cc[cc], (lv.link, m)) do
          return zeros(C, dim(sites[n]), dim(sites[n]))
        end

        add_to_local_matrix!(matrix_element, weight, local_op, lv.needs_JW_string)
      end
    end

    ## Q and prow are no longer needed.
    Q = nothing
    prow = nothing

    ## If we are at the last site, then R will be a 1x1 matrix containing an overall scaling.
    if n == length(sites)
      @assert nccs == 1
      
      if left_size(ccs, cc) == 1
        scaling = only(g.edges_from_left[lv_id])[2]
      else
        scaling = only(R)
      end

      for block in values(matrix_of_cc[cc])
        block .*= scaling
      end

      ## We can the also skip building the next graph.
      continue
    end

    ## Build the graph for the next site out of this component.
    next_edges = Matrix{Vector{Tuple{Int, C}}}(undef, rank, length(op_cache_vec[n + 1]))
    for i in eachindex(next_edges)
      next_edges[i] = Vector{Tuple{Int, C}}()
    end

    ## A specialization for when there is only one vertex on the left.
    if left_size(ccs, cc) == 1
      build_next_edges_specialization!(next_edges, g, n, g.edges_from_left[lv_id])

      empty!(g.edges_from_left[lv_id])
      sizehint!(g.edges_from_left[lv_id], 0)
    else
      for_non_zeros_batch(R, length(right_map)) do weights, ms, j
        ## Convert j, which has been permuted first by the connected components
        ## and then again by SPQR into a right vertex Id.
        rv_id = right_map[pcol[j]]

        ## Get the operator acting on site (n + 1) of this right vertex.
        op_id = get_onsite_op(right_vertex(g, rv_id), n + 1)

        ## Find the first equivalent right vertex from site (n + 2) and onward.
        ## Going forward, this will be the active vertex.
        rv_id = find_first_eq_rv(g, rv_id, n + 2)

        ## Add the edges.
        for (weight, m) in zip(weights, ms)
          m > rank && return
          push!(next_edges[m, op_id], (rv_id, weight))
        end
      end
    end

    next_edges_of_cc[cc] = next_edges
  end

  ## If we are merging the connected components by the quantum numbers.
  cc_order = [i for i in 1:nccs]
  if combine_qn_sectors && has_qns
    cc_order, qi_of_cc = merge_qn_sectors(qi_of_cc)
  end

  ## Combine the graphs of each component together
  next_graph = MPOGraph{N, C, Ti}([], g.right_vertices, [])
  offset_of_cc = zeros(Int, nccs + 1)

  cur_offset = 0
  for cc in cc_order
    offset_of_cc[cc] = cur_offset
    add_to_next_graph!(next_graph, g, op_cache_vec, n, cur_offset, next_edges_of_cc[cc])
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
