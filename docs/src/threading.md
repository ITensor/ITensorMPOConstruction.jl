# Threading and performance

A plurality (if not majority) of the time spent in MPO construction is in performing the sparse QR decomposition. This can often be sped-up significantly by loading MKL.

```julia
try
  using MKL
catch
end
```

ITensorMPOConstruction can take advantage of multiple threads in a variety of ways.

1. `ITensorMPS.add!(os::OpIDSum, ...)` is thread-safe. This can be used to construct a `OpIDSum` in parallel.

2. During the MPO construction `Threads.@threads` are used, primarily to iterate over the connected components (i.e. QN blocks) in parallel.

3. In the sparse QR decomposition that is done for every connected component. The threading here is controlled by `BLAS.set_num_threads`. Over subscription is possible since this QR decomposition is itself called from within `Threads.@threads`.

## Threading tips

* The MPO construction is memory bound, and the performance gain (or loss!) from multi-threading is system and operator dependent.
* If the MPO has lots of connected components (i.e. is sparse), use Julia threads.
* If the MPO has few connected components (i.e. is dense) use BLAS threads.

The transcorrelated momentum space Fermi-Hubbard MPO (see [Challenge Problem](./examples/fermi-hubbard-tc.md)) is very sparse. As such, giving Julia all the threads results in the best performance. The following timings are for the ``8 \times 8`` system on a computer with two Intel(R) Xeon(R) Gold 6438Y+ (64 total threads) and 250 GiB of memory.

| Julia threads | BLAS Threads | OpIDSum time | MPO time |
|---------------|--------------|--------------|----------|
| 1             | 1            | 203s         | 2796s    |
| 1             | 64           | 203s         | 2582s    |
| 64            | 1            | 33s          | 1426s    |
| 64            | 64           | 33s          | 3010s    |

