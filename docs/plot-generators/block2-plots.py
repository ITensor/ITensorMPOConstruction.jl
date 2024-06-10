import numpy as np
from timeit import default_timer as timer
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

L = 100
N = L
TWOSZ = 0

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=L, n_elec=N, spin=TWOSZ)

t = 1
U = 4

b = driver.expr_builder()

epsilon = lambda k : 2 * np.cos(2 * np.pi * k / L)

# Kinetic term
for k in range(L):
  b.add_term("cd", np.array([k, k]), -t * epsilon(k))
  b.add_term("CD", np.array([k, k]), -t * epsilon(k))

# Interaction term
sites = np.zeros(4 * L * L * L, dtype=np.int16)
for p in range(L):
  for q in range(L):
    for k in range(L):
      b.add_term("cCDd", np.array([(p - k) % L, (q + k) % L, q, p]), U)

print("Done constructing representation.")

alg = MPOAlgorithmTypes.SVD | MPOAlgorithmTypes.Blocked
# alg = MPOAlgorithmTypes.SVD | MPOAlgorithmTypes.Fast | MPOAlgorithmTypes.Blocked | MPOAlgorithmTypes.Disjoint
start = timer()
mpo = driver.get_mpo(b.finalize(), iprint=1, algo_type=alg)
stop = timer()
print(f"N = {L}, time = {stop - start}")