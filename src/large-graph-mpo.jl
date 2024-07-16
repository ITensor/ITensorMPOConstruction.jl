BlockSparseMatrix{C} = Dict{Tuple{Int,Int},Matrix{C}}

MPOGraph{N, C} = BipartiteGraph{LeftVertex, NTuple{N, OpID}, C}

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


@timeit function MPOGraph(os::OpIDSum{N, C}, op_cache_vec::OpCacheVec)::MPOGraph{N, C} where {N, C}
  for i in 1:length(os)
    for j in size(os.terms, 1):-1:1
      if os.terms[j, i] != zero(OpID)
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
    if os.scalars[i] != 0
      nnz += 1
      os.scalars[nnz] = os.scalars[i]
      os._data[nnz] = os._data[i]
    end
  end

  os.num_terms = nnz
  resize!(os._data, nnz)
  resize!(os.scalars, nnz)

  g = MPOGraph{N, C}([], os._data, [Vector{Tuple{Int, C}}() for _ in 1:nnz])
  for (op_id, op_info) in enumerate(op_cache_vec[1])
    push!(g.left_vertices, LeftVertex(1, op_id, op_info.is_fermionic))
  end

  for j in 1:right_size(g)
    weight = os.scalars[j]
    onsite_op_id = get_onsite_op_id(right_vertex(g, j), 1)
    push!(g.edges_from_right[j], (onsite_op_id, weight))
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
  if tol < 0
    ret = qr(A)
  else
    ret = qr(A; tol=tol)
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

@timeit function at_site!(
  ValType::Type{<:Number},
  g::MPOGraph{N, C},
  left_link_is_fermionic::Vector{Bool},
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  op_cache_vec::OpCacheVec;
  output_level::Int=0)::Tuple{MPOGraph{N, C}, Vector{Bool}, Vector{Int},Vector{BlockSparseMatrix{ValType}},Index} where {N, C}

  has_qns = hasqns(sites)
  
  combine_duplicate_adjacent_right_vertices!(g, terms_eq_from(n + 1))
  ccs = compute_connected_components!(g)
  nccs = num_connected_components(ccs)

  output_level > 0 && println("  The graph has $(left_size(g)) left vertices, $(right_size(g)) right vertices, $(num_edges(g)) edges and $(nccs) connected components")

  next_graph = MPOGraph{N, C}([], g.right_vertices, Vector{Tuple{Int, C}}[Vector{Tuple{Int, C}}() for _ in 1:right_size(g)])

  offset_of_cc = zeros(Int, nccs + 1)
  matrix_of_cc = [BlockSparseMatrix{ValType}() for _ in 1:nccs]
  qi_of_cc = Pair{QN,Int}[QN() => 0 for _ in 1:nccs]

  right_link_is_fermionic = Vector{Bool}()

  @timeit "Threaded loop" for cc in 1:nccs
    W, left_map, right_map = get_cc_matrix(g, ccs, cc)

    ## This information is now in W and not needed again.
    clear_edges!(g, right_map)

    ## Compute the decomposition and then free W
    Q, R, prow, pcol, rank = sparse_qr(W, tol)
    W = nothing

    ## TODO: This isn't thread safe
    offset_of_cc[cc + 1] = offset_of_cc[cc] + rank

    if has_qns
      right_flux = flux(right_vertex(g, right_map[1]), n + 1, op_cache_vec)
      qi_of_cc[cc] = (QN() - right_flux) => rank
    end

    # Form the local transformation tensor.
    let
      for_non_zeros_batch(Q, rank) do weights, m
        first_time = true
        for (i, weight) in enumerate(weights)
          weight == 0 && continue

          # Convert the linear index into the 2D index (link, op_id).
          lv_id = left_map[prow[i]]
          link = (lv_id - 1) ÷ length(op_cache_vec[n]) + 1
          op_id = lv_id - (link - 1) * length(op_cache_vec[n])

          local_op = op_cache_vec[n][op_id].matrix

          matrix_element = get!(matrix_of_cc[cc], (link, m)) do
            return zeros(C, dim(sites[n]), dim(sites[n]))
          end

          needs_JW_string = xor(left_link_is_fermionic[link], op_cache_vec[n][op_id].is_fermionic)
          if first_time
            first_time = false
            push!(right_link_is_fermionic, needs_JW_string)
          end

          add_to_local_matrix!(matrix_element, weight, local_op, needs_JW_string)
        end
      end
    end

    ## If we are at the last site, then R will be a 1x1 matrix containing an overall scaling.
    if n == length(sites)
      scaling = only(R)
      for block in values(matrix_of_cc[cc])
        block .*= scaling
      end
    end

    # Build the graph for the next site. If we are at the last site then we can skip this step.
    ## TODO: This isn't thread safe
    ## TODO: Pretty sure if this seems faster we can make it even better by storing some is_fermionic info, but maybe not.
    if n != length(sites)
      for m in 1:rank
        for op_id in 1:length(op_cache_vec[n + 1])
          push!(next_graph.left_vertices, LeftVertex(0, 0, false))
        end
      end

      for_non_zeros_batch(R, length(right_map)) do weights, ms, j
        j = right_map[pcol[j]]
        op_id = get_onsite_op_id(right_vertex(g, j), n + 1)

        end_idx = searchsortedlast(ms, rank)
        resize!(next_graph.edges_from_right[j], end_idx)

        for idx in 1:end_idx
          weight, m = weights[idx], ms[idx]
          lv_id = length(op_cache_vec[n + 1]) * (m + offset_of_cc[cc] - 1) + op_id
          next_graph.edges_from_right[j][idx] = (lv_id, weight)
        end
      end
    end
  end

  if has_qns
    outgoing_link = Index(qi_of_cc...; tags="Link,l=$n", dir=ITensors.Out)
  else
    outgoing_link = Index(offset_of_cc[end]; tags="Link,l=$n")
  end

  return next_graph, right_link_is_fermionic, offset_of_cc, matrix_of_cc, outgoing_link
end
