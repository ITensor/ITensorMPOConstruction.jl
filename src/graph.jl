struct LeftVertex
  link::Int32
  opOnSite::OpID
  needs_JW_string::Bool
end

function Base.hash(v::LeftVertex, h::UInt)
  return hash((v.link, v.opOnSite), h)
end

function Base.:(==)(v1::LeftVertex, v2::LeftVertex)::Bool
  return (v1.link, v1.opOnSite) == (v2.link, v2.opOnSite)
end

struct RightVertex
  ops::Vector{OpID}
  is_fermionic::Bool
end

function Base.hash(v::RightVertex, h::UInt)
  return hash(v.ops, h)
end

function Base.:(==)(v1::RightVertex, v2::RightVertex)::Bool
  return v1.ops == v2.ops
end

struct MPOGraph{C}
  edge_left_vertex::Vector{Int}
  edge_right_vertex::Vector{Int}
  edge_weight::Vector{C}
  left_vertex_ids::Dict{LeftVertex,Int}
  right_vertex_ids::Dict{RightVertex,Int}
  left_vertex_values::Vector{LeftVertex}
  right_vertex_values::Vector{RightVertex}
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

  push!(g.left_vertex_values, lv)

  for i in eachindex(os)
    scalar, ops = os[i]

    scalar == 0 && continue

    rv = RightVertex(reverse(ops), false)
    push!(g.right_vertex_values, rv)

    push!(g.edge_left_vertex, 1)
    push!(g.edge_right_vertex, length(g.right_vertex_values))
    push!(g.edge_weight, scalar)
  end

  return g
end

function Base.empty!(g::MPOGraph)::Nothing
  empty!(g.edge_left_vertex)
  empty!(g.edge_right_vertex)
  empty!(g.edge_weight)
  empty!(g.left_vertex_values)
  empty!(g.right_vertex_values)

  return nothing
end

function left_size(g::MPOGraph)::Int
  return length(g.left_vertex_values)
end

function right_size(g::MPOGraph)::Int
  return length(g.right_vertex_values)
end

function num_edges(g::MPOGraph)::Int
  return length(g.edge_left_vertex)
end

function left_value(g::MPOGraph{C}, left_id::Int)::LeftVertex where {C}
  return g.left_vertex_values[left_id]
end

function right_value(g::MPOGraph{C}, right_id::Int)::RightVertex where {C}
  return g.right_vertex_values[right_id]
end

function add_edges!(
  g::MPOGraph{C},
  rv::RightVertex,
  rank::Int,
  ms::AbstractVector{Int},
  m_offset::Int,
  onsite_op::OpID,
  weights::AbstractVector{C},
)::Nothing where {C}
  right_id = get!(g.right_vertex_ids, rv) do
    push!(g.right_vertex_values, rv)
    return length(g.right_vertex_values)
  end

  for i in 1:length(ms)
    ms[i] > rank && return nothing

    lv = LeftVertex(ms[i] + m_offset, onsite_op, rv.is_fermionic)

    left_id = get!(g.left_vertex_ids, lv) do
      push!(g.left_vertex_values, lv)
      return length(g.left_vertex_values)
    end

    push!(g.edge_left_vertex, left_id)
    push!(g.edge_right_vertex, right_id)
    push!(g.edge_weight, weights[i])
  end

  return nothing
end

function add_edges_vector_lookup!(
  g::MPOGraph{C},
  rv::RightVertex,
  rank::Int,
  ms::AbstractVector{Int},
  m_offset::Int,
  onsite_op::OpID,
  weights::AbstractVector{C},
  id_of_left_vertex::Vector{Int},
)::Nothing where {C}
  right_id = get!(g.right_vertex_ids, rv) do
    push!(g.right_vertex_values, rv)
    return length(g.right_vertex_values)
  end

  for i in 1:length(ms)
    m = ms[i]
    m > rank && return nothing

    if id_of_left_vertex[m] == 0
      push!(g.left_vertex_values, LeftVertex(m + m_offset, onsite_op, rv.is_fermionic))
      id_of_left_vertex[m] = length(g.left_vertex_values)
    end

    push!(g.edge_left_vertex, id_of_left_vertex[m])
    push!(g.edge_right_vertex, right_id)
    push!(g.edge_weight, weights[i])
  end

  return nothing
end

@timeit function sparse_edge_weights(g::MPOGraph{C})::SparseMatrixCSC{C,Int} where {C}
  @assert length(g.edge_left_vertex) == length(g.edge_right_vertex)
  @assert length(g.edge_left_vertex) == length(g.edge_weight)

  return sparse(g.edge_left_vertex, g.edge_right_vertex, g.edge_weight)
end
