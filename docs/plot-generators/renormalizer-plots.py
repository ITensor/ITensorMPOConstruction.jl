import numpy as np
from timeit import default_timer as timer
from renormalizer import Model, Op, BasisSimpleElectron, Mpo


def fermiHubbardKS(N, t, U):
    toSite = lambda k, s: 2 * k + s

    terms = []
    for k in range(N):
        for spin in (0, 1):
            factor = -2 * t * np.cos((2 * np.pi * k) / N)
            terms.append(Op(r"a^\dagger a", toSite(k, spin), factor))
    
    for p in range(N):
        for q in range(N):
            for k in range(N):
                factor = U / N
                sites = [toSite((p - k) % N, 1), toSite((q + k) % N, 0), toSite(q, 0), toSite(p, 1)]
                terms.append(Op(r"a^\dagger a^\dagger a a", sites, factor))
    
    model = Model([BasisSimpleElectron(i) for i in range(2 * N)], terms)
    return Mpo(model)

numSites = []
bondDims = []
times = []
for N in [10, 20, 30, 40, 50]:
  start = timer()
  mpo = fermiHubbardKS(N, 1, 4)
  stop = timer()
  
  numSites.append(N)
  bondDims.append(max(mpo.bond_dims))
  times.append(stop - start)

  print(f"N = {N}, time = {times[-1]}")

print("numSites = ", numSites)
print("bondDims = ", bondDims)
print("times = ", times)
