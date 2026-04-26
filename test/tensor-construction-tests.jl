using ITensorMPOConstruction
using ITensors
using Test

function reference_to_ITensor(
  offsets::Vector{Int},
  block_sparse_matrices::Vector{ITensorMPOConstruction.BlockSparseMatrix{C}},
  llink::ITensors.QNIndex,
  rlink::ITensors.QNIndex,
  site::ITensors.QNIndex;
  splitblocks=false,
  tol=0.0,
  checkflux=true,
)::ITensor where {C}
  if splitblocks
    llink = ITensors.splitblocks(llink)
    rlink = ITensors.splitblocks(rlink)
    site = ITensors.splitblocks(site)
  end

  T = ITensors.BlockSparseTensor(
    C,
    Block{4}[],
    (dag(llink), rlink, prime(site), dag(site)),
  )

  for (offset, matrix) in zip(offsets, block_sparse_matrices)
    for (right_link, column) in enumerate(matrix)
      for (left_link, block) in column
        for i in axes(block, 1)
          for j in axes(block, 2)
            if abs(block[i, j]) > tol
              T[left_link, right_link + offset, i, j] = block[i, j]
            end
          end
        end
      end
    end
  end

  checkflux && ITensors.checkflux(T)
  return itensor(T)
end

function test_sparse_to_ITensor_matches_reference(splitblocks::Bool=false)
  llink = Index([QN("N", 0) => 2, QN("N", 1) => 1]; tags="Link,l=1", dir=ITensors.Out)
  rlink = Index([QN("N", 0) => 1, QN("N", 1) => 2]; tags="Link,l=2", dir=ITensors.Out)
  site = Index([QN("N", 0) => 2, QN("N", 1) => 1]; tags="Site", dir=ITensors.Out)

  offsets = [0, 1]
  block_sparse_matrices = [
    [
      Dict(1 => [1.0 -2.0 0.0; 3.0 4.0 0.0; 0.0 0.0 5.0]),
      Dict(3 => [0.0 0.0 0.0; -6.0 0.0 0.0; 0.0 0.0 -1.5]),
    ],
    [
      Dict(2 => [0.0 0.0 7.0; 0.0 0.0 -8.0; 0.0 0.0 0.0]),
      Dict(3 => [0.25 0.5 0.0; -0.75 1.0 0.0; 0.0 0.0 -2.0]),
    ],
  ]

  expected = reference_to_ITensor(
    offsets, block_sparse_matrices, llink, rlink, site; splitblocks
  )
  actual = ITensorMPOConstruction.to_ITensor(
    offsets, block_sparse_matrices, llink, rlink, site; splitblocks
  )

  @test actual == expected
  @test nnz(actual) == nnz(expected)
end

function test_sparse_to_ITensor_all_zero_blocks(splitblocks::Bool=false)
  llink = Index([QN("N", 0) => 2, QN("N", 1) => 1]; tags="Link,l=1", dir=ITensors.Out)
  rlink = Index([QN("N", 0) => 1, QN("N", 1) => 2]; tags="Link,l=2", dir=ITensors.Out)
  site = Index([QN("N", 0) => 2, QN("N", 1) => 1]; tags="Site", dir=ITensors.Out)

  offsets = [0]
  block_sparse_matrices = [[Dict(1 => zeros(ComplexF64, dim(site), dim(site)))]]

  expected = reference_to_ITensor(
    offsets, block_sparse_matrices, llink, rlink, site; splitblocks
  )
  actual = ITensorMPOConstruction.to_ITensor(
    offsets, block_sparse_matrices, llink, rlink, site; splitblocks
  )

  @test actual == expected
  @test nnz(actual) == 0
end

@testset "Tensor Construction" begin
  for splitblocks in (false, true)
    test_sparse_to_ITensor_matches_reference(splitblocks)
    test_sparse_to_ITensor_all_zero_blocks(splitblocks)
  end
end
