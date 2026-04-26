# `OpIDSum`

For some systems, constructing the `OpSum` can be the bottleneck of MPO construction. `OpIDSum` plays the same role as `OpSum`, but in a much more efficient manner. To specify an operator in a term of an `OpSum`, you provide a string (the operator's name) and a site index. To specify an operator in a term of an `OpIDSum`, you provide an `OpID`, which contains an operator index and a site. The operator index is the position of the operator in the provided basis for the site. The remaining metadata about the operator is kept in an `OpCacheVec`.

```@docs
OpID
OpIDSum
ITensorMPS.add!
OpInfo
OpCacheVec
to_OpCacheVec
Base.length
Base.eachindex
Base.getindex
Base.zero
Base.isless
ITensors.flux
```
