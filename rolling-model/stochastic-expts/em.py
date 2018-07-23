# Python analog of em.m in Higham, 2001
# Euler-Maruyama method on linear SDE

# SDE is dW = lam*X dt + mu*X dW, X(0) = Xzero
#   where lam = 2, mu = 1, and Xzero = 1.

# Discretized Brownian path over [0, 1] has dt = 2**(-8).
# Euler-Maruyama uses timestep R*dt

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(100)
lam, mu, Xzero = 2, 1, 1
T, N = 1, 2**8
dt = 1/N
t = np.linspace(start=0, stop=T, num=N+1)
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
W = np.insert(arr=np.cumsum(dW), obj=0, values=0)

Xtrue = Xzero*np.exp((lam - mu**2/2)*t + mu*W)  # Compute the true path
plt.plot(t, Xtrue, 'm-')

R = 1; Dt = R*dt; L = N//R
Xem = np.zeros(shape=L+1)
Xtemp = Xzero
Xem[0] = Xtemp
for j in range(0, L):
    Winc = np.sum(dW[R*j:R*(j+1)])
    Xtemp += Dt*lam*Xtemp + mu*Xtemp*Winc
    Xem[j+1] = Xtemp

plt.plot(np.linspace(start=0, stop=T, num=N/R+1), Xem, 'r--*')
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.show()

emerr = np.abs(Xem[-1] - Xtrue[-1])
print(emerr)
