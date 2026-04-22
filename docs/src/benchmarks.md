# Benchmarks: Momentum Space Fermi-Hubbard

We constructed the 1D momentum space Fermi-Hubbard Hamiltonian using ITensorMPS and ITensorMPOConstruction. For even $N$, the Hamiltonian can be represented exactly as an MPO of bond dimension $10 N - 4$, and both algorithms achieve this minimal bond dimension. ITensorMPOConstruction is also not only able to construct this particular MPO much faster, but the sparsity of the resulting MPO is much higher. These timings were taken with `julia -t8 --gcthreads=8,1` on a 2021 MacBook Pro with the M1 Max CPU and 32GB of memory. The ITensorMPS data is from [`fermi-hubbard-ks.jl`](examples/fermi-hubbard-ks.md), and the ITensorMPOConstruction data is from [`fermi-hubbard-tc.jl`](examples/fermi-hubbard-tc.md) with equivalent settings.

| $N$ | ITensorMPS    | ITensorMPOConstruction  |
|-----|---------------|-------------------------|
| 10  | 0.32s / 92.7% | 0.01s / 99.32%          |
| 20  | 30.6s / 92.6% | 0.06s / 99.70%          |
| 30  | 792s / 92.6%  | 0.53s / 99.81%          |
| 40  | N/A           | 0.55s / 99.86%          |
| 50  | N/A           | 0.38s / 99.89%          |
| 100 | N/A           | 2.81s / 99.94%          |
| 200 | N/A           | 30.3s / 99.97%          |
| 300 | N/A           | 136s / 99.982%          |
| 400 | N/A           | 415s / 99.986%          |
| 500 | N/A           | 1274s / 99.989%         |
