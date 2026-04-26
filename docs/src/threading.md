# Threading and performance

For the QR decomposition, much of the time spent in MPO construction is in the sparse QR decomposition. This can often be sped up significantly by loading MKL.

```julia
try
  using MKL
catch
end
```

ITensorMPOConstruction can take advantage of multiple threads in a variety of ways.

1. `ITensorMPS.add!(os::OpIDSum, ...)` is thread-safe. This can be used to construct an `OpIDSum` in parallel.

2. During MPO construction, `Threads.@threads` is used primarily to iterate over the connected components (i.e. QN blocks) in parallel.

3. With `alg="QR"`, sparse QR decompositions are performed for connected components. The threading here is controlled by `BLAS.set_num_threads`. Oversubscription is possible since this QR decomposition is itself called from within `Threads.@threads`.

## Threading tips

* MPO construction is memory-bound, and the performance gain (or loss!) from multithreading is system and operator dependent.
* Change the number of threads used for [garbage collection](https://docs.julialang.org/en/v1/manual/memory-management/#man-gc-multithreading).
* If the MPO has lots of connected components (i.e. is sparse), use Julia threads.
* If the MPO has few connected components (i.e. is dense), try using BLAS threads.
