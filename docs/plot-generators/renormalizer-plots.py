import numpy as np
from timeit import default_timer as timer
from renormalizer import Model, Op, BasisSimpleElectron, Mpo
import random


def fermi_hubbard_ks(N, t, U):
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


def electronic_structure(N):
    toSite = lambda k, s: 2 * k + s
    coeff = lambda : random.random()

    terms = []
    for a in range(N):
        for b in range(a, N):
            factor = coeff()
            for spin in (0, 1):
                sites = [toSite(a, spin), toSite(b, spin)]
                terms.append(Op(r"a^\dagger a", sites, factor))

                if a != b:
                    sites = [toSite(b, spin), toSite(a, spin)]
                    terms.append(Op(r"a^\dagger a", sites, np.conj(factor)))
    
    for j in range(N):
        for s_j in (0, 1):

            for k in range(N):
                s_k = s_j
                if (s_k, k) <= (s_j, j):
                    continue

                for l in range(N):
                    for s_l in (0, 1):
                        if (s_l, l) <= (s_j, j):
                            continue

                        for m in range(N):
                            s_m = s_l
                            if (s_m, m) <= (s_k, k):
                                continue

                            value = coeff()
                            sites = [toSite(j, s_j), toSite(l, s_l), toSite(m, s_m), toSite(k, s_k)]
                            terms.append(Op(r"a^\dagger a^\dagger a a", sites, factor))
                            
                            sites = list(reversed(sites))
                            terms.append(Op(r"a^\dagger a^\dagger a a", sites, np.conj(factor)))

    model = Model([BasisSimpleElectron(i) for i in range(2 * N)], terms)
    return Mpo(model)


numSites = []
bondDims = []
times = []
for N in [40]:
  start = timer()
  mpo = electronic_structure(N)
  stop = timer()
  
  numSites.append(N)
  bondDims.append(max(mpo.bond_dims))
  times.append(stop - start)

  print(f"N = {N}, time = {times[-1]}")

print("numSites = ", numSites)
print("bondDims = ", bondDims)
print("times = ", times)
