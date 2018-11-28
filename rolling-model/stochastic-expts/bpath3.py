# Python analog of bpath3.m in Higham, 2001
# Function along a Brownian path

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(100)

T, N = 1., 500
dt = T/N
t = np.linspace(start=0, stop=T, num=N+1)

M = 1000
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(M, N))
W = np.insert(arr=np.cumsum(dW, axis=1), obj=0, values=0, axis=1)
U = np.exp(np.tile(t, reps=(M, 1)) + W/2)
U_mean = np.mean(U, axis=0)

plt.plot(t, U_mean, 'b-')
plt.plot(t, np.transpose(U[:100, :]), 'r--', linewidth=.3)
plt.xlabel('$t$')
plt.ylabel('$U(t)$')
plt.legend(['mean of 1000 paths', '100 individual paths'])
plt.show()

averr = np.linalg.norm((U_mean - np.exp(9*t/8)), ord=np.inf)
print(averr)
