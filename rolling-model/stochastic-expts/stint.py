# Python analog of stint.m in Higham, 2001
# Ito and Stratonovich integrals of W dW

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(100)
T, N = 1, 500
dt = T/N

dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
W = np.insert(arr=np.cumsum(dW), obj=0, values=0)

ito = np.sum(W[:-1]*dW)
strat = np.sum(((W[:-1] + W[1:])/2 +
                np.random.normal(loc=0, scale=np.sqrt(dt),
                                 size=N))*dW)

itoerr = np.abs(ito - (W[-1]**2 - T)/2)
straterr = np.abs(strat - W[-1]**2/2)

print(ito, strat)
print(itoerr, straterr)
