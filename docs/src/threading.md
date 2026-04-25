# Threading and performance

For the QR decomposition, a plurality (if not majority) of the time spent in MPO construction is in performing the sparse QR decomposition. This can often be sped-up significantly by loading MKL.

```julia
try
  using MKL
catch
end
```

ITensorMPOConstruction can take advantage of multiple threads in a variety of ways.

1. `ITensorMPS.add!(os::OpIDSum, ...)` is thread-safe. This can be used to construct a `OpIDSum` in parallel.

2. During the MPO construction `Threads.@threads` are used, primarily to iterate over the connected components (i.e. QN blocks) in parallel.

3. With `alg=QR`, in the sparse QR decomposition that is done for every connected component. The threading here is controlled by `BLAS.set_num_threads`. Over subscription is possible since this QR decomposition is itself called from within `Threads.@threads`.

## Threading tips

* The MPO construction is memory bound, and the performance gain (or loss!) from multi-threading is system and operator dependent.
* Change the number of threads used for [garbage collection](https://docs.julialang.org/en/v1/manual/memory-management/#man-gc-multithreading).
* If the MPO has lots of connected components (i.e. is sparse), use Julia threads.
* If the MPO has few connected components (i.e. is dense) try using BLAS threads.