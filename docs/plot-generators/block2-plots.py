import numpy as np
from timeit import default_timer as timer
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

L = 16
N = L
TWOSZ = 0

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=L, n_elec=N, spin=TWOSZ)

t = 1
U = 4
J = -0.5

b = driver.expr_builder()

epsilon = lambda k : 2 * np.cos(2 * np.pi * k / L)

# Kinetic term
for k in range(L):
  b.add_term("cd", np.array([k, k]), -t * epsilon(k))
  b.add_term("CD", np.array([k, k]), -t * epsilon(k))

# Interaction term
for p in range(L):
  for q in range(L):
    for k in range(L):
      if J == 0:
        b.add_term("cCDd", np.array([(p - k) % L, (q + k) % L, q, p]), U / N)
      else:
        factor = U - t * ((np.exp(J) - 1) * epsilon(p - k) + (np.exp(-J) - 1) * epsilon(p))
        b.add_term("cCDd", np.array([(p - k) % L, (q + k) % L, q, p]), factor / N)
        b.add_term("CcdD", np.array([(p - k) % L, (q + k) % L, q, p]), factor / N)

if J != 0:
  prefactor = 2 * t * (np.cosh(J) - 1) / N**2
  for p in range(L):
    for q in range(L):
      for s in range(L):
        for k in range(L):
          for kp in range(L):
            factor = prefactor * epsilon(p - (k - kp))
            b.add_term("cCCDDd", np.array([(p - k) % L, (q + kp) % L, (s + k - kp) % L, s, q, p]), factor)
            b.add_term("CccddD", np.array([(p - k) % L, (q + kp) % L, (s + k - kp) % L, s, q, p]), factor)

print("Done constructing representation.")

# alg = MPOAlgorithmTypes.SVD | MPOAlgorithmTypes.Blocked
alg = MPOAlgorithmTypes.SVD | MPOAlgorithmTypes.Fast | MPOAlgorithmTypes.Blocked | MPOAlgorithmTypes.Disjoint
start = timer()
mpo = driver.get_mpo(b.finalize(), iprint=1, algo_type=alg)
stop = timer()
print(f"N = {L}, time = {stop - start}")