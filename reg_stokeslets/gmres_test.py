# Test of the GMRES solver

import numpy as np
from matrix_routines import gmres

a = np.random.rand(10, 10)
b = np.random.rand(10)

x = gmres(a)

print(np.linalg.norm(a @ x - b))
