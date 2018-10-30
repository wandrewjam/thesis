# Code for some useful matrix routines not included in the SciPy package

import numpy as np
from scipy.linalg import norm


def arnoldi(a, b=None, n_top=None):

    if a.ndim != 2:
        print('A must have exactly 2 dimensions')
        return None, None

    m = a.shape[0]
    if m != a.shape[1]:
        print('A must be a square matrix')
        return None, None

    if b is None:
        b = np.random.rand(m)

    if n_top is None:
        n_top = m - 1

    q = np.zeros(shape=(m, n_top+1))
    q[:, 0] = b/norm(b)
    h = np.zeros(shape=(n_top+1, n_top))

    for n in range(n_top):
        qcurr = q[:, n]
        v = a @ qcurr  # Multiply a*q_1
        for j in range(n+1):
            qj = q[:, j]
            h[j, n] = np.dot(qj, v)
            v -= h[j, n]*qj
        h[n+1, n] = norm(v)
        if h[n+1, n] == 0:
            q[:, n+1] = np.random.rand(m)
        else:
            q[:, n+1] = v/h[n+1, n]
    return q, h


# Function to solve the linear system ax = b for x using GMRES
def gmres(a, b=None, eps=1e-6, max_iter=None):

    if eps <= 0:
        print('eps must be a positive number')

    if a.ndim != 2:
        print('A must have exactly 2 dimensions')
        return None, None

    m = a.shape[0]
    if m != a.shape[1]:
        print('A must be a square matrix')
        return None, None

    if b is None:
        b = np.random.rand(m)

    if max_iter is None:
        max_iter = np.minimum(m-1, 100)

    q = np.zeros(shape=(m, 1))
    q[:, 0] = b/norm(b)
    h = np.zeros(shape=(1, 0))
    n = 0
    err = eps + 1

    x = np.zeros(shape=(m, 0))

    while err > eps and n < max_iter:
        h = np.concatenate((h, np.zeros(shape=(n+1, 1))), axis=1)
        qcurr = q[:, n]
        v = a @ qcurr  # Multiply a*q_1
        for j in range(n+1):
            qj = q[:, j]
            h[j, n] = np.dot(qj, v)
            v -= h[j, n]*qj
        h = np.concatenate((h, np.zeros(shape=(1, n+1))), axis=0)
        h[n+1, n] = norm(v)
        if h[n+1, n] == 0:
            q = np.concatenate((q, np.random.rand(m, 1)), axis=1)
        else:
            q = np.concatenate((q, v[:, None]/h[n+1, n]), axis=1)

        # Canonical basis vector for use in the GMRES iteration
        e1 = np.zeros(shape=(n+2, 1))
        e1[0] = 1

        y = np.linalg.lstsq(h, norm(b)*e1, rcond=None)[0]
        x = np.concatenate((x, q[:, :-1] @ y), axis=1)

        err = norm(a @ x[:, -1] - b)

        n += 1

    return x
