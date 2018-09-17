from sampling import *
import numpy as np
import matplotlib.pyplot as plt
from time_dependent import *

N = 200
bond_max = 10
eta_om = .01


data_v = np.load('v_ssa_N{:d}_bmax{:d}_eta{:g}.npz'.format(N, bond_max, eta_om))
data_t = np.load('t_ssa_N{:d}_bmax{:d}_eta{:g}.npz'.format(N, bond_max, eta_om))
data_fixed1 = np.load('alg1_N{:d}_bmax{:d}_eta{:g}.npz'.format(N, bond_max, eta_om))
data_fixed2 = np.load('alg2_N{:d}_bmax{:d}_eta{:g}.npz'.format(N, bond_max, eta_om))

det_v, _, _, det_t = time_dependent()
det_v2, _, _, det_t2 = time_dependent(saturation=False)

samples = 1000
v_samp = np.zeros(shape=(100, samples))
t_samp = np.linspace(start=0, stop=0.4, num=samples)
i = 0
for key in data_t.keys():
    t, v = data_t[key], data_v[key]
    v_samp[i, :] = sample_velocity(t, v, t_samp)
    i += 1
v3_avg, v3_std = np.mean(v_samp, axis=0), np.std(v_samp, axis=0)
t1, v = data_fixed1['t1'], data_fixed1['v1']
v1_avg, v1_std = np.mean(v, axis=0), np.std(v, axis=0)
t2, v = data_fixed2['t2'], data_fixed2['v2']
v2_avg, v2_std = np.mean(v, axis=0), np.std(v, axis=0)

# Plotting code
fig, ax = plt.subplots()
ax.plot(t_samp, v3_avg, 'r-', linewidth=.5, label='Variable Step')
ax.plot(t_samp[1:], (v3_avg+v3_std)[1:], 'r--', linewidth=.5)
ax.plot(t_samp[1:], (v3_avg-v3_std)[1:], 'r--', linewidth=.5)

ax.plot(t1, v1_avg, 'b-', linewidth=.5, label='Fixed Step')
ax.plot(t1[1:], (v1_avg+v1_std)[1:], 'b--', linewidth=.5)
ax.plot(t1[1:], (v1_avg-v1_std)[1:], 'b--', linewidth=.5)

ax.plot(t2, v2_avg, 'g-', linewidth=.5, label='Fixed Step, no Saturation')
ax.plot(t2[1:], (v2_avg+v2_std)[1:], 'g--', linewidth=.5)
ax.plot(t2[1:], (v2_avg-v2_std)[1:], 'g--', linewidth=.5)

ax.plot(det_t, det_v, 'k', linewidth=1.5, label='PDE Model')
ax.plot(det_t2, det_v2, 'k-.', linewidth=1.5, label='PDE Model, no Saturation')
ax.plot(t_samp, 22*np.ones(shape=t_samp.shape), 'k--', linewidth=1, label='$V_f$')

ax.legend(loc=1)
ax.set_title('Comparison of stochastic algorithms with PDE model')
ax.set_xlabel('Nondimensional time')
ax.set_ylabel('Nondimensional linear velocity')

plt.show()
