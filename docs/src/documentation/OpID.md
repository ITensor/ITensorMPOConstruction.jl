# `OpIDSum`

For some systems constructing the `OpSum` can actually be the bottleneck of MPO construction. `OpIDSum` plays the same roll as `OpSum` but in a much more efficient manner. To specify an operator in a term of an `OpSum` you specify a string (the operator's name) and a site index, whereas to specify an operator in a term of an `OpIDSum` you specify an `OpID` which contains an operator index and a site. The operator index is the index of the operator in the provided basis for the site. The remainder of the information about what the operator actually is, is kept in an `OpCacheVec`.

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
ITensors.flux
```