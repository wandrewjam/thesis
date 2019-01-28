# Code to simulate the simple SSA example from Gillespie 2007

import numpy as np
import matplotlib.pyplot as plt

# Parameters
c, x0 = 1, 1000


def rre(t):
    return x0*np.exp(-c*t)


def cme_mean(t):
    return rre(t)


def cme_sdev(t):
    return np.sqrt(rre(t)*(1 - np.exp(-c*t)))


def ssa(t_max=10):
    # The simple Gillespie SSA algorithm

    x_vec = -np.ones(x0+1, dtype=int)
    t_vec = -np.ones(x0+1)
    x_vec[0], t_vec[0] = x0, 0

    r = np.random.rand(x0)
    counter = 0
    while t_vec[counter] < t_max and x_vec[counter] != 0:
        tau = np.log(1/r[counter])/(c*x_vec[counter])
        x_vec[counter+1] = x_vec[counter] - 1
        t_vec[counter+1] = t_vec[counter] + tau
        counter += 1

    return t_vec, x_vec


num_expts = 3
ssa_results = []
for i in range(num_expts):
    ssa_results.append(ssa())

t_high = 0
for i in range(num_expts):
    t_expt_max = np.max(ssa_results[i][0])
    if t_high < t_expt_max:
        t_high = t_expt_max

t = np.linspace(0, t_high, num=200)
rre_result = rre(t)
cme_result = cme_sdev(t)

for i in range(num_expts):
    plt.step(ssa_results[i][0], ssa_results[i][1], where='post')
plt.plot(t, rre_result, 'k-', color='black', linewidth=1)
plt.plot(t[1:], rre_result[1:] - 2*cme_result[1:], color='grey', linestyle='dashed')
plt.plot(t[1:], rre_result[1:] + 2*cme_result[1:], color='grey', linestyle='dashed')
plt.show()
