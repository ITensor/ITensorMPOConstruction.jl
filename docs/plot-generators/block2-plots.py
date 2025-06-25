import numpy as np
from timeit import default_timer as timer
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import random

def fermi_hubbard(N, alg, t=1, U=4, J=0):
    start = timer()
    
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=N, n_elec=N, spin=0)

    b = driver.expr_builder()

    epsilon = lambda k : 2 * np.cos(2 * np.pi * k / N)

    # Kinetic term
    for k in range(N):
        b.add_term("cd", np.array([k, k]), -t * epsilon(k))
        b.add_term("CD", np.array([k, k]), -t * epsilon(k))

    # Interaction term
    for p in range(N):
        for q in range(N):
            for k in range(N):
                if J == 0:
                    b.add_term("cCDd", np.array([(p - k) % N, (q + k) % N, q, p]), U / N)
                else:
                    factor = U - t * ((np.exp(J) - 1) * epsilon(p - k) + (np.exp(-J) - 1) * epsilon(p))
                    b.add_term("cCDd", np.array([(p - k) % N, (q + k) % N, q, p]), factor / N)
                    b.add_term("CcdD", np.array([(p - k) % N, (q + k) % N, q, p]), factor / N)

    if J != 0:
        prefactor = 2 * t * (np.cosh(J) - 1) / N**2
        for p in range(N):
            for q in range(N):
                for s in range(N):
                    for k in range(N):
                        for kp in range(N):
                            factor = prefactor * epsilon(p - (k - kp))
                            b.add_term("cCCDDd", np.array([(p - k) % N, (q + kp) % N, (s + k - kp) % N, s, q, p]), factor)
                            b.add_term("CccddD", np.array([(p - k) % N, (q + kp) % N, (s + k - kp) % N, s, q, p]), factor)

    print("Done constructing representation.")

    b = b.finalize()
    print("Done finalizing representation.")

    mpo = driver.get_mpo(b, iprint=1, algo_type=alg)
    stop = timer()
    print(f"N = {N}, time = {stop - start}")


def get_coefficients(N):
    rng = np.random.default_rng(0)
    one_electron = rng.normal(size=(N, N))
    two_electron = rng.normal(size=(N, 2, N, 2, N, 2, N, 2))

    return one_electron, two_electron

def electronic_structure(N, alg):
    one_electron_coeffs, two_electron_coeffs = get_coefficients(N)    

    start = timer()
    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=8)
    driver.initialize_system(n_sites=N, n_elec=N, spin=0, orb_sym=None)

    builder = driver.expr_builder()

    c = lambda spin : "c" if (spin == 0) else "C"
    d = lambda spin : "d" if (spin == 0) else "D"

    for a in range(N):
        for b in range(a, N):
            factor = one_electron_coeffs[a, b]
            for spin in (0, 1):
                sites = np.array([a, b])
                builder.add_term(c(spin) + d(spin), sites, factor)

                if a != b:
                    sites = [b, a]
                    builder.add_term(c(spin) + d(spin), sites, factor)

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

                            factor = two_electron_coeffs[j, s_j, l, s_l, m, s_m, k, s_k]
                            sites = np.array([j, l, m, k])
                            builder.add_term(c(s_j) + c(s_l) + d(s_m) + d(s_k), sites, factor)
                            
                            sites = np.flip(sites)
                            builder.add_term(c(s_k) + c(s_m) + d(s_l) + d(s_j), sites, factor)

    print("Done constructing representation.")

    builder = builder.finalize()
    print("Done finalizing representation.")

    mpo = driver.get_mpo(builder, iprint=1, algo_type=alg)
    stop = timer()
    print(f"N = {N}, time = {stop - start}")


for N in [10]:
    fermi_hubbard(N, MPOAlgorithmTypes.FastBlockedDisjointSVD)
    print()

for N in [10]:
    electronic_structure(N, MPOAlgorithmTypes.FastBipartite)
    print()
