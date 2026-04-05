# Benchmarks: Fermi-Hubbard Hamiltonian in Momentum Space

We constructed the momentum space Fermi-Hubbard Hamiltonian using ITensorMPS and ITensorMPOConstruction. For even $N$, the Hamiltonian can be represented exactly as an MPO of bond dimension $10 N - 4$, and both algorithms achieve this minimal bond dimension. ITensorMPOConstruction is also not only able to construct this particular MPO much faster, but the sparsity of the resulting MPO is much higher. These timings were taken with `julia -t8` on a 2021 MacBook Pro with the M1 Max CPU and 32GB of memory. The ITensorMPS data is from `fermi-hubbard-ks.jl`, and the ITensorMPOConstruction data is from `fermi-hubbard-tc.jl` with equivalent settings.

## Bond Dimension 
| $N$ | ITensorMPS | ITensorMPOConstruction |
|-----|------------|------------------------|
| 10  | 96         | 96                     |
| 20  | 196        | 196                    |
| 30  | 296        | 296                    |
| 40  | N/A        | 396                    |
| 50  | N/A        | 496                    |
| 100 | N/A        | 996                    |
| 200 | N/A        | 1996                   |
| 300 | N/A        | 2996                   |
| 400 | N/A        | 3996                   |
| 500 | N/A        | 4996                   |

## Sparsity 
Sparsity of the `ITensorMPS` MPO with the default `splitblocks=true`, and the `ITensorMPOConstruction` MPO with the less aggressive `combine_qn_sectors::Bool=false`.

| $N$ | ITensorMPS | ITensorMPOConstruction |
|-----|------------|------------------------|
| 10  | 92.7%      | 99.32%                 |
| 20  | 92.6%      | 99.70%                 |
| 30  | 92.6%      | 99.81%                 |
| 40  | N/A        | 99.86%                 |
| 50  | N/A        | 99.89%                 |
| 100 | N/A        | 99.94%                 |
| 200 | N/A        | 99.97%                 |
| 300 | N/A        | 99.982%                |
| 400 | N/A        | 99.986%                |
| 500 | N/A        | 99.989%                |

## Runtime 
| $N$ | ITensorMPS | ITensorMPOConstruction |
|-----|------------|------------------------|
| 10  | 0.32s      | 0.01s                  |
| 20  | 30.6s      | 0.03s                  |
| 30  | 792s       | 0.10s                  |
| 40  | N/A        | 0.20s                  |
| 50  | N/A        | 0.32s                  |
| 100 | N/A        | 2.84s                  |
| 200 | N/A        | 33.5s                  |
| 300 | N/A        | 155s                   |
| 400 | N/A        | 512s                   |
| 500 | N/A        | 1352s                  |
