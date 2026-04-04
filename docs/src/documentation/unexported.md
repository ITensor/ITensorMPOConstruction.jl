# Things you might want to use

```@docs
ITensorMPOConstruction.resume_MPO_construction
ITensorMPOConstruction.sparsity
ITensorMPOConstruction.block2_nnz
```

# Things you probably don't want to use

## Ops

```@docs
ITensorMPOConstruction.is_fermionic
ITensorMPOConstruction.are_equal
ITensorMPOConstruction.sort_fermion_perm!
ITensorMPOConstruction.determine_val_type
ITensorMPOConstruction.for_equal_sites
ITensorMPOConstruction.rewrite_in_operator_basis!
ITensorMPOConstruction.op_sum_to_opID_sum
ITensorMPOConstruction.check_os_for_errors
ITensorMPOConstruction.prepare_opID_sum!
ITensorMPOConstruction.get_onsite_op
```

## Bipartite Graph

```@docs
  ITensorMPOConstruction.BipartiteGraph
  ITensorMPOConstruction.BipartiteGraphConnectedComponents
  ITensorMPOConstruction.left_size
  ITensorMPOConstruction.right_size
  ITensorMPOConstruction.left_vertex
  ITensorMPOConstruction.right_vertex
  ITensorMPOConstruction.num_edges
  ITensorMPOConstruction.compute_connected_components
  ITensorMPOConstruction.num_connected_components
  ITensorMPOConstruction.get_cc_matrix
```

## Others

```@docs
ITensorMPOConstruction.BlockSparseMatrix
ITensorMPOConstruction.add_to_local_matrix!
ITensorMPOConstruction.CoSorterElement
ITensorMPOConstruction.sparse_qr
ITensorMPOConstruction.my_ITensor
ITensorMPOConstruction.LeftVertex
ITensorMPOConstruction.CoSorter
ITensorMPOConstruction.add_to_next_graph!
ITensorMPOConstruction.MPOGraph
ITensorMPOConstruction.find_first_eq_rv
ITensorMPOConstruction.for_non_zeros_batch
ITensorMPOConstruction.merge_qn_sectors
ITensorMPOConstruction.@time_if
ITensorMPOConstruction.at_site!
ITensorMPOConstruction.build_next_edges_specialization!
```