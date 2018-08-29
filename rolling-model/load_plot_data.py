from sampling import *
import numpy as np
import matplotlib.pyplot as plt
from time_dependent import *

bond_max = 10


data_v = np.load('variable_experiment_v.npz')
data_t = np.load('variable_experiment_t.npz')
data_fixed1 = np.load('fixed_experiment_1.npz')
data_fixed2 = np.load('fixed_experiment_2.npz')

det_v, _, det_t, _ = time_dependent()
det_v2, _, det_t2, _ = time_dependent(saturation=False)

# plt.plot(data_t['arr_0'], 22*data_t['arr_0'], 'k--', linewidth=1)
# for key in data_t.keys():
#     t, v = data_t[key], data_v[key]
#     plt.plot(t[1:], np.cumsum(v[1:]*(t[1:]-t[:-1])), 'b', linewidth=.1)
# t, v = data_fixed['t1'], data_fixed['v1']
# plt.plot(t[1:], np.transpose(np.cumsum(
#     v[:, 1:]*(t[1:]-t[:-1]), axis=1)), 'r', linewidth=.1)
# plt.plot(det_t[1:], np.transpose(np.cumsum(
#     det_v[1:]*(det_t[1:]-det_t[:-1]))), 'k', linewidth=1.5)
# plt.show()

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
plt.plot(t_samp, v3_avg, 'r-', t_samp[1:], (v3_avg+v3_std)[1:], 'r--',
         t_samp[1:], (v3_avg-v3_std)[1:], 'r--', linewidth=.5)
plt.plot(t1, v1_avg, 'b-', t1[1:], (v1_avg+v1_std)[1:], 'b--',
         t1[1:], (v1_avg-v1_std)[1:], 'b--', linewidth=.5)
plt.plot(t2, v2_avg, 'g-', t2[1:], (v2_avg+v2_std)[1:], 'g--',
         t2[1:], (v2_avg-v2_std)[1:], 'g--', linewidth=.5)
plt.plot(det_t, det_v, 'k', linewidth=1.5)
plt.plot(det_t2, det_v2, 'k-.')
plt.plot(data_t['arr_0'], 22*np.ones(shape=data_t['arr_0'].shape), 'k--', linewidth=1)
plt.show()
