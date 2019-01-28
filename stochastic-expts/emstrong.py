# Python analog of emstrong.m in Higham, 2001
# Test strong convergence of Euler-Maruyama

# Solves dX = lam*X dt + mu*X dW, X(0) = Xzero
#   where lam = 2, mu = 1, and Xzero = 1.

# Discretized Brownian path over [0, 1] has dt = 2**(-9).
# E-M uses 5 different timesteps: 16dt, 8dt, 4dt, 2dt, dt.
# Examine strong convergence at T=1: E | X_L - X(T) |.

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(100)
lam, mu, Xzero = 2., 1., 1.
T, N = 1., 2**9
dt = T/N
M = 1000

Xerr = np.zeros(shape=(M, 5))
for s in range(M):
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
    W = np.insert(arr=np.cumsum(dW), obj=0, values=0)
    Xtrue = Xzero*np.exp((lam - mu**2/2) + mu*W[-1])  # Compute the true path
    for p in range(5):
        R = 2**p
        Dt = R*dt
        L = int(N/R)
        Xtemp = Xzero
        for j in range(L):
            Winc = np.sum(dW[R*j:R*(j+1)])
            Xtemp += Dt*lam*Xtemp + mu*Xtemp*Winc
        Xerr[s, p] = np.abs(Xtemp - Xtrue)

Dtvals = dt*(2**np.arange(5))
fig, ax = plt.subplot(nrows=2, ncols=2, sharex='all', sharey='all')
plt.loglog(Dtvals, np.mean(Xerr, axis=0), 'b*-')
plt.loglog(Dtvals, np.sqrt(Dtvals), 'r--')
plt.axis([1e-3, 1e-1, 1e-4, 1])
plt.xlabel('$\\Delta t$')
plt.ylabel('Sample average of $| X(T) - X_L |$')
plt.title('emstrong.m')
plt.show()

# Least squares fit of error = C * Dt**q #
