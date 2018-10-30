from sampling import sample_velocity
import numpy as np
import matplotlib.pyplot as plt
from time_dependent import time_dependent

N = 256
bond_max = 10
eta_om = .01
gamma = 20.0
num_iterations = 100

L = 2.5
M = 256
time_steps = 2000
z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1), np.linspace(-np.pi/2, np.pi/2, N+1),
                              indexing='ij')


data_v = np.load('./data/v_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}.npz'.format(N, bond_max, eta_om, gamma))
data_n = np.load('./data/n_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}.npz'.format(N, bond_max, eta_om, gamma))
data_t = np.load('./data/t_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}.npz'.format(N, bond_max, eta_om, gamma))
data_fixed1 = np.load('./data/alg1_N{:d}_bmax{:d}_eta{:g}_gamma{:g}.npz'.format(N, bond_max, eta_om, gamma))
data_fixed2 = np.load('./data/alg2_N{:d}_bmax{:d}_eta{:g}_gamma{:g}.npz'.format(N, bond_max, eta_om, gamma))

det_v, _, m_mesh, det_t = time_dependent(M=M, N=N, time_steps=time_steps, eta_om=eta_om, eta_v=eta_om, gamma=gamma,
                                         save_m=True)
det_v2, _, m_mesh2, det_t2 = time_dependent(M=M, N=N, time_steps=time_steps, eta_om=eta_om, eta_v=eta_om, gamma=gamma,
                                            saturation=False, save_m=True)

det_n = np.trapz(np.trapz(m_mesh, z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)
det_n2 = np.trapz(np.trapz(m_mesh2, z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)

samples = time_steps
v_samp = np.zeros(shape=(num_iterations, samples))
n_samp = np.zeros(shape=(num_iterations, samples))
t_samp = np.linspace(start=0, stop=0.4, num=samples)
i = 0
for key in data_t.keys():
    t, v, n = data_t[key], data_v[key], data_n[key]
    v_samp[i, :] = sample_velocity(t, v, t_samp)
    n_samp[i, :] = sample_velocity(t, n, t_samp)
    i += 1
v3_avg, v3_std = np.mean(v_samp, axis=0), np.std(v_samp, axis=0)
n3_avg, n3_std = np.mean(n_samp, axis=0), np.std(n_samp, axis=0)
t1, v, n = data_fixed1['t1'], data_fixed1['v1'], data_fixed1['n1']
v1_avg, v1_std = np.mean(v, axis=0), np.std(v, axis=0)
n1_avg, n1_std = np.mean(n, axis=0), np.std(n, axis=0)
t2, v, n = data_fixed2['t2'], data_fixed2['v2'], data_fixed2['n2']
v2_avg, v2_std = np.mean(v, axis=0), np.std(v, axis=0)
n2_avg, n2_std = np.mean(n, axis=0), np.std(n, axis=0)

# Plotting code for velocity
fig, ax = plt.subplots(nrows=3, sharex='col', figsize=(8, 9))
ax[0].plot(t_samp, v3_avg, 'r-', linewidth=.5, label='Variable Step')
ax[0].plot(t_samp[1:], (v3_avg+2*v3_std/np.sqrt(num_iterations))[1:], 'r:', linewidth=.5)
ax[0].plot(t_samp[1:], (v3_avg-2*v3_std/np.sqrt(num_iterations))[1:], 'r:', linewidth=.5)

ax[1].plot(t1, v1_avg, 'b-', linewidth=.5, label='Fixed Step')
ax[1].plot(t1[1:], (v1_avg+2*v1_std/np.sqrt(num_iterations))[1:], 'b:', linewidth=.5)
ax[1].plot(t1[1:], (v1_avg-2*v1_std/np.sqrt(num_iterations))[1:], 'b:', linewidth=.5)

ax[2].plot(t2, v2_avg, 'g-', linewidth=.5, label='Fixed Step, no Saturation')
ax[2].plot(t2[1:], (v2_avg+2*v2_std/np.sqrt(num_iterations))[1:], 'g:', linewidth=.5)
ax[2].plot(t2[1:], (v2_avg-2*v2_std/np.sqrt(num_iterations))[1:], 'g:', linewidth=.5)

for i in range(2):
    ax[i].plot(det_t, det_v, 'k', linewidth=1.5, label='PDE Model')
ax[2].plot(det_t2, det_v2, 'k-.', linewidth=1.5, label='PDE Model, no Saturation')

ax[0].set_title('Comparison of stochastic algorithms with PDE model')
for i in range(3):
    ax[i].plot(t_samp, 1.1*gamma*np.ones(shape=t_samp.shape), 'k--', linewidth=1, label='$V_f$')
    ax[i].legend(loc='best')
    ax[i].set_ylabel('Nondimensional linear velocity')

ax[2].set_xlabel('Nondimensional time')
plt.tight_layout()
plt.show()

# Plotting code for bond quantity
sat = N*bond_max/np.pi

fig, ax = plt.subplots(nrows=3, sharex='col', figsize=(8, 9))
ax[0].plot(t_samp, n3_avg/sat, 'r-', linewidth=.5, label='Variable Step')
ax[0].plot(t_samp[1:], (n3_avg+2*n3_std/np.sqrt(num_iterations))[1:]/sat, 'r:', linewidth=.5)
ax[0].plot(t_samp[1:], (n3_avg-2*n3_std/np.sqrt(num_iterations))[1:]/sat, 'r:', linewidth=.5)

ax[1].plot(t1, n1_avg/sat, 'b-', linewidth=.5, label='Fixed Step')
ax[1].plot(t1[1:], (n1_avg+2*n1_std/np.sqrt(num_iterations))[1:]/sat, 'b:', linewidth=.5)
ax[1].plot(t1[1:], (n1_avg-2*n1_std/np.sqrt(num_iterations))[1:]/sat, 'b:', linewidth=.5)

ax[2].plot(t2, n2_avg/sat, 'g-', linewidth=.5, label='Fixed Step, no Saturation')
ax[2].plot(t2[1:], (n2_avg+2*n2_std/np.sqrt(num_iterations))[1:]/sat, 'g:', linewidth=.5)
ax[2].plot(t2[1:], (n2_avg-2*n2_std/np.sqrt(num_iterations))[1:]/sat, 'g:', linewidth=.5)

for i in range(2):
    ax[i].plot(det_t, det_n, 'k', linewidth=1.5, label='PDE Model')
ax[2].plot(det_t2, det_n2, 'k-.', linewidth=1.5, label='PDE Model, no Saturation')

ax[0].set_title('Comparison of stochastic algorithms with PDE model')
for i in range(3):
    ax[i].legend(loc='best')
    ax[i].set_ylabel('Nondimensional bond quantity')

ax[2].set_xlabel('Nondimensional time')
plt.tight_layout()
plt.show()
