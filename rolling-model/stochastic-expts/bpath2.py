# Python analog of bpath2.m in Higham, 2001
# Brownian path simulation

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(100)

T, N = 1., 500
dt = T/N

dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
W = np.insert(arr=np.cumsum(dW), obj=0, values=0)

plt.plot(np.linspace(start=0, stop=T, num=N+1), W, 'r-')
plt.xlabel('$t$')
plt.ylabel('$W(t)$')
plt.show()
