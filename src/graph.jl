struct LeftVertex
  link::Int32
  opOnSite::OpID
  needsJWString::Bool
end

function Base.hash(v::LeftVertex, h::UInt)
  return hash((v.link, v.opOnSite), h)
end

function Base.:(==)(v1::LeftVertex, v2::LeftVertex)::Bool
  return (v1.link, v1.opOnSite) == (v2.link, v2.opOnSite)
end

struct RightVertex
  ops::Vector{OpID}
  isFermionic::Bool
end

function Base.hash(v::RightVertex, h::UInt)
  return hash(v.ops, h)
end

function Base.:(==)(v1::RightVertex, v2::RightVertex)::Bool
  return v1.ops == v2.ops
end

struct MPOGraph{C}
  edgeLeftVertex::Vector{Int}
  edgeRightVertex::Vector{Int}
  edgeWeight::Vector{C}
  leftVertexIDs::Dict{LeftVertex,Int}
  rightVertexIDs::Dict{RightVertex,Int}
  leftVertexValues::Vector{LeftVertex}
  rightVertexValues::Vector{RightVertex}
end

function MPOGraph{C}() where {C}
  return MPOGraph(
    Vector{Int}(),
    Vector{Int}(),
    Vector{C}(),
    Dict{LeftVertex,Int}(),
    Dict{RightVertex,Int}(),
    Vector{LeftVertex}(),
    Vector{RightVertex}(),
  )
end

"""
Construct the graph of the sum split about a dummy site 0.

Every term in the sum must be unique and sorted by site, have 0 flux, and even fermionic parity.
"""
@timeit function MPOGraph{C}(os::OpIDSum) where {C}
  g = MPOGraph{C}()
  lv = LeftVertex(0, OpID(0, 0), false)

  push!(g.leftVertexValues, lv)

  for i in eachindex(os)
    scalar, ops = os[i]

    scalar == 0 && continue

    rv = RightVertex(reverse(ops), false)
    push!(g.rightVertexValues, rv)

    push!(g.edgeLeftVertex, 1)
    push!(g.edgeRightVertex, length(g.rightVertexValues))
    push!(g.edgeWeight, scalar)
  end

  return g
end

function Base.empty!(g::MPOGraph)::Nothing
  empty!(g.edgeLeftVertex)
  empty!(g.edgeRightVertex)
  empty!(g.edgeWeight)
  empty!(g.leftVertexValues)
  empty!(g.rightVertexValues)

  return nothing
end

function left_size(g::MPOGraph)::Int
  return length(g.leftVertexValues)
end

function right_size(g::MPOGraph)::Int
  return length(g.rightVertexValues)
end

function num_edges(g::MPOGraph)::Int
  return length(g.edgeLeftVertex)
end

function left_value(g::MPOGraph{C}, leftID::Int)::LeftVertex where {C}
  return g.leftVertexValues[leftID]
end

function right_value(g::MPOGraph{C}, rightID::Int)::RightVertex where {C}
  return g.rightVertexValues[rightID]
end

function add_edges!(
  g::MPOGraph{C},
  rv::RightVertex,
  rank::Int,
  ms::AbstractVector{Int},
  mOffset::Int,
  onsiteOp::OpID,
  weights::AbstractVector{C},
)::Nothing where {C}
  rightID = get!(g.rightVertexIDs, rv) do
    push!(g.rightVertexValues, rv)
    return length(g.rightVertexValues)
  end

  for i in 1:length(ms)
    ms[i] > rank && return nothing

    lv = LeftVertex(ms[i] + mOffset, onsiteOp, rv.isFermionic)

    leftID = get!(g.leftVertexIDs, lv) do
      push!(g.leftVertexValues, lv)
      return length(g.leftVertexValues)
    end

    push!(g.edgeLeftVertex, leftID)
    push!(g.edgeRightVertex, rightID)
    push!(g.edgeWeight, weights[i])
  end

  return nothing
end

function add_edges_id!(
  g::MPOGraph{C},
  rv::RightVertex,
  rank::Int,
  ms::AbstractVector{Int},
  mOffset::Int,
  onsiteOp::OpID,
  weights::AbstractVector{C},
  fooBar::Vector{Int},
)::Nothing where {C}
  rightID = get!(g.rightVertexIDs, rv) do
    push!(g.rightVertexValues, rv)
    return length(g.rightVertexValues)
  end

  for i in 1:length(ms)
    m = ms[i]
    m > rank && return nothing

    if fooBar[m] == 0
      push!(g.leftVertexValues, LeftVertex(m + mOffset, onsiteOp, rv.isFermionic))
      fooBar[m] = length(g.leftVertexValues)
    end

    push!(g.edgeLeftVertex, fooBar[m])
    push!(g.edgeRightVertex, rightID)
    push!(g.edgeWeight, weights[i])
  end

  return nothing
end

@timeit function sparse_edge_weights(g::MPOGraph{C})::SparseMatrixCSC{C,Int} where {C}
  @assert length(g.edgeLeftVertex) == length(g.edgeRightVertex)
  @assert length(g.edgeLeftVertex) == length(g.edgeWeight)

  return sparse(g.edgeLeftVertex, g.edgeRightVertex, g.edgeWeight)
end
