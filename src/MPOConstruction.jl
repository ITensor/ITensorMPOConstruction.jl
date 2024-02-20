BlockSparseMatrix{C} = Dict{Tuple{Int,Int},Matrix{C}}

@timeit function sparse_qr(
  A::SparseMatrixCSC, tol::Real
)::Tuple{SparseMatrixCSC,SparseMatrixCSC,Vector{Int},Vector{Int},Int}
  if tol < 0
    ret = qr(A)
  else
    ret = qr(A; tol=tol)
  end

  return sparse(ret.Q), ret.R, ret.prow, ret.pcol, rank(ret)
end

function for_non_zeros(f::Function, A::SparseMatrixCSC, maxRow::Int, maxCol::Int)::Nothing
  @assert maxRow <= size(A, 1)
  @assert maxCol <= size(A, 2)

  rows = rowvals(A)
  vals = nonzeros(A)
  for col in 1:maxCol
    for idx in nzrange(A, col)
      row = rows[idx]
      if row <= maxRow
        value = vals[idx]
        f(value, row, col)
      end
    end
  end
end

function for_non_zeros_batch(f::Function, A::SparseMatrixCSC, maxCol::Int)::Nothing
  @assert maxCol <= size(A, 2) "$maxCol, $(size(A, 2))"

  rows = rowvals(A)
  vals = nonzeros(A)
  for col in 1:maxCol
    range = nzrange(A, col)
    isempty(range) && continue
    f((@view vals[range]), (@view rows[range]), col)
  end
end

function for_non_zeros_batch(f::Function, A::SparseMatrixCSC, cols::Vector{Int})::Nothing
  rows = rowvals(A)
  vals = nonzeros(A)
  for col in cols
    @assert 0 < col <= size(A, 2)
    range = nzrange(A, col)
    isempty(range) && continue
    f((@view vals[range]), (@view rows[range]), col)
  end
end

@timeit function at_site!(
  graphs::Dict{QN,MPOGraph{C}},
  n::Int,
  sites::Vector{<:Index},
  tol::Real,
  opCacheVec::OpCacheVec,
)::Tuple{Dict{QN,MPOGraph{C}},BlockSparseMatrix{C},Index} where {C}
  hasQNs = hasqns(sites)
  nextGraphs = Dict{QN,MPOGraph{C}}()

  matrix = BlockSparseMatrix{C}()

  qi = Vector{Pair{QN,Int}}()
  outgoingLinkOffset = 0

  mIdentityToID = Vector{Int}()

  for (inQN, g) in graphs
    W = sparse_edge_weights(g)
    Q, R, prow, pcol, rank = sparse_qr(W, tol)

    W = nothing

    #=
    If we are at the last site, then R will be a 1x1 matrix containing an overall scaling factor
    that we need to account for.
    =#
    if n == length(sites)
      Q *= only(R)
    end

    if hasQNs
      append!(qi, [inQN => rank])
    end

    # Form the local transformation tensor.
    # At the 0th dummy site we don't need to do this.
    @timeit "constructing the sparse matrix" if n != 0
      ## TODO: This is wastefull
      localMatrices = [
        my_matrix([lv.opOnSite], opCacheVec[n]; needsJWString=lv.needsJWString) for
        lv in g.leftVertexValues
      ]

      for_non_zeros(Q, left_size(g), rank) do weight, i, m
        rightLink = m + outgoingLinkOffset
        lv = left_value(g, prow[i])

        matrixElement = get!(matrix, (lv.link, rightLink)) do
          return zeros(C, dim(sites[n]), dim(sites[n]))
        end

        matrixElement .+= weight * localMatrices[prow[i]]
      end
    end

    ## Free up some memory
    Q = prow = nothing

    # Connect this output \tilde{A}_m to \tilde{B}_m = (R \vec{B})_m
    # If we are processing the last site then there's no need to do this.
    @timeit "constructing the next graphs" if n != length(sites)
      nextGraphOfZeroFlux = get!(nextGraphs, inQN) do
        return MPOGraph{C}()
      end

      resize!(mIdentityToID, rank)
      mIdentityToID .= 0
      for_non_zeros_batch(R, right_size(g)) do weights, ms, j
        j = pcol[j]

        rv = right_value(g, j)
        if !isempty(rv.ops) && rv.ops[end].n == n + 1
          onsite = pop!(rv.ops)
          flux = opCacheVec[n + 1][onsite.id].qnFlux
          newIsFermionic = xor(rv.isFermionic, opCacheVec[n + 1][onsite.id].isFermionic)

          rv = RightVertex(rv.ops, newIsFermionic)

          nextGraph = get!(nextGraphs, inQN + flux) do
            return MPOGraph{C}()
          end

          add_edges!(nextGraph, rv, rank, ms, outgoingLinkOffset, onsite, weights)
        else
          add_edges_id!(
            nextGraphOfZeroFlux,
            rv,
            rank,
            ms,
            outgoingLinkOffset,
            OpID(1, n + 1),
            weights,
            mIdentityToID,
          )
        end
      end

      if num_edges(nextGraphOfZeroFlux) == 0
        delete!(nextGraphs, inQN)
      end

      for (_, nextGraph) in nextGraphs
        empty!(nextGraph.leftVertexIDs)
      end
    end

    empty!(g)
    outgoingLinkOffset += rank
  end

  for (_, nextGraph) in nextGraphs
    empty!(nextGraph.rightVertexIDs)
  end

  if hasQNs
    outgoingLink = Index(qi...; tags="Link,l=$n", dir=ITensors.Out)
  else
    outgoingLink = Index(outgoingLinkOffset; tags="Link,l=$n")
  end

  return nextGraphs, matrix, outgoingLink
end

@timeit function svdMPO_new(
  ValType::Type{<:Number},
  os::OpIDSum{C},
  opCacheVec::OpCacheVec,
  sites::Vector{<:Index};
  tol::Real=-1,
)::MPO where {C}
  # TODO: This should be fixed.
  @assert !ITensors.using_auto_fermion()

  N = length(sites)

  graphs = Dict{QN,MPOGraph{C}}(QN() => MPOGraph{C}(os))

  H = MPO(sites)

  llinks = Vector{Index}(undef, N + 1)
  for n in 0:N
    graphs, symbolicMatrix, llinks[n + 1] = at_site!(graphs, n, sites, tol, opCacheVec)

    # For the 0th iteration we only care about constructing the graphs for the next site.
    n == 0 && continue

    # Constructing the tensor from an array is much faster than setting the components of the ITensor directly.
    # Especially for sparse tensors, and works for both sparse and dense tensors.
    @timeit "creating ITensor" let
      tensor = zeros(
        ValType,
        dim(dag(llinks[n])),
        dim(llinks[n + 1]),
        dim(prime(sites[n])),
        dim(dag(sites[n])),
      )

      for ((leftLink, rightLink), localOpMatrix) in symbolicMatrix
        tensor[leftLink, rightLink, :, :] = localOpMatrix
      end

      H[n] = itensor(
        tensor,
        dag(llinks[n]),
        llinks[n + 1],
        prime(sites[n]),
        dag(sites[n]);
        tol=1e-10,
        checkflux=false,
      )
    end
  end

  # Remove the dummy link indices from the left and right.
  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(dag(llinks[N + 1]))
  R[1] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end

function svdMPO_new(os::OpIDSum{C}, opCacheVec::OpCacheVec, sites; kwargs...)::MPO where {C}
  # Function barrier to improve type stability
  ValType = determine_val_type(os)
  return svdMPO_new(ValType, os, opCacheVec, sites; kwargs...)
end

function MPO_new(
  os::OpIDSum,
  sites::Vector{<:Index},
  opCacheVec::OpCacheVec;
  basisOpCacheVec::Union{Nothing,OpCacheVec}=nothing,
  kwargs...,
)::MPO
  os, opCacheVec = prepare_opID_sum!(os, sites, opCacheVec, basisOpCacheVec)

  return svdMPO_new(os, opCacheVec, sites; kwargs...)
end

function MPO_new(os::OpSum, sites::Vector{<:Index}; kwargs...)::MPO
  opIDSum, opCacheVec = op_sum_to_opID_sum(os, sites)
  return MPO_new(opIDSum, sites, opCacheVec; kwargs...)
end

function sparsity(mpo::MPO)::Float64
  numEntries = 0
  numZeros = 0
  for tensor in mpo
    numEntries += prod(size(tensor))
    numZeros += prod(size(tensor)) - ITensors.nnz(tensor)
  end

  return numZeros / numEntries
end
