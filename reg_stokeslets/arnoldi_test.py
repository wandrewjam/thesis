# Test of the Arnoldi iteration
import numpy as np
from matrix_routines import arnoldi

N = 100
a = np.random.rand(N, N)
q, h = arnoldi(a, n_top=N//100)

print(np.linalg.norm(a @ q[:, :-1] - q @ h))
