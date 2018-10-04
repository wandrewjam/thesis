import numpy as np
import matplotlib.pyplot as plt
from time_dependent import time_dependent
from constructA import length

L, T = 2.5, 0.4
M, N, time_steps = 100, 100, 1000
d_prime = 0.1
eta, delta, kap = 0.1, 3.0, 1.0

n = 4
gamma_max = 20
Kd = 0.02
eta_om, eta_v = 0.01, 0.01

t = np.linspace(start=0, stop=T, num=time_steps+1)
gamma = gamma_max*t**n/(t**n + Kd**n)
v, om, m_mesh, t = time_dependent(expt='unmoving', L=L, T=T, time_steps=time_steps,
                                  d_prime=d_prime, eta=eta, delta=delta, kap=kap, eta_v=eta_v,
                                  eta_om=eta_om, gamma=gamma)

z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+2)[1:],
                              np.linspace(-np.pi/2, np.pi/2, N+2)[1:],
                              indexing='ij')
l_matrix = length(z_mesh, th_mesh, d_prime)
form_rate = kap*np.exp(-eta/2*l_matrix**2)
break_rate = np.exp(delta*l_matrix)

# Total bonds
total_bonds = np.trapz(np.trapz(m_mesh[:-1, :-1, :], z_mesh[:-1, 0], axis=0), th_mesh[0, :-1], axis=0)

# Calculate average bond length
avg_bond_length = np.trapz(np.trapz(l_matrix[:-1, :-1, None]*m_mesh[:-1, :-1, :], z_mesh[:-1, 0],
                                    axis=0), th_mesh[0, :-1], axis=0)/total_bonds

# Calculate total formation rate
avg_form_rate = np.trapz(np.trapz(form_rate[:-1, :-1, None] *
                                  (1 - np.trapz(m_mesh[:-1, :-1, :], z_mesh[:-1, 0], axis=0))[None, :, :],
                                  z_mesh[:-1, 0], axis=0), th_mesh[0, :-1], axis=0)

# Calculate total breaking rate
avg_break_rate = np.trapz(np.trapz(break_rate[:-1, :-1, None]*m_mesh[:-1, :-1, :], z_mesh[:-1, 0], axis=0),
                          th_mesh[0, :-1], axis=0)
z_breaking = v*np.trapz(m_mesh[0, :-1, :], th_mesh[0, :-1], axis=0)
th_breaking = om*np.trapz(m_mesh[:-1, 0, :], z_mesh[:-1, 0], axis=0)

# Calculate force and torque
force = eta_v*(v - (1 + d_prime)*gamma)
torque = eta_om*(om - gamma)

# Plotting code
fig, ax = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 9.5))
# fig.set_size_inches(8.5, 8.5)

ax[0, 0].plot(t, v, t, om)
ax[0, 0].plot(t, (1 + d_prime)*gamma, 'k--')
# ax[0].plot(t, (1 + d_prime)*gamma, 'k--')
ax[0, 0].set_ylabel('Nondimensional angular or linear velocity')
ax[0, 0].set_title('Velocities of stably bound platelet')
ax[0, 0].legend(['Linear velocity', 'Angular velocity'], loc='best')

# ax1 = ax[0].twinx()
# ax1.plot(t, om, 'r')
# ax1.set_ylabel('Nondimensional angular velocity')

# ax[1].plot(t, avg_bond_length)
ax[1, 0].plot(t[1:-1], (total_bonds[2:] - total_bonds[:-2])/(t[1] - t[0]))
ax[1, 0].set_ylabel('Nondimensional bond number per time')
ax[1, 0].set_title('Derivative of bond quantity')

ax[2, 1].plot(t, avg_bond_length, t, total_bonds)
ax[2, 1].set_ylabel('Nondimensional length or bond quantity')
ax[2, 1].set_title('Average bond length and number of bonds')
ax[2, 1].legend(['Average bond length', 'Quantity of bonds'], loc='best')

# ax1 = ax[1].twinx()
# ax1.plot(t, total_bonds, 'r')

# ax[2].plot(t, avg_form_rate, t, avg_break_rate + z_breaking + th_breaking)
ax[2, 0].plot(t, avg_form_rate, t, avg_break_rate)
ax[2, 0].set_ylabel('Nondimensional rate ($1/s$)')
ax[2, 0].set_title('Total formation and breaking rates')
ax[2, 0].legend(['Total formation rate', 'Total breaking rate'], loc='best')

# ax2 = ax[2].twinx()
# ax2.plot(t, avg_break_rate, 'r')
# ax2.set_ylabel('Nondimensional rate ($1/s$)')
# ax2.set_legend(['Formation rate', 'Breaking rate'])

ax[2, 0].set_xlabel('Nondimensional time ($s$)')

ax[0, 1].plot(t, z_breaking, t, th_breaking)
ax[0, 1].set_ylabel('Nondimensional rate ($1/s$)')
ax[0, 1].set_title('Bond breaking through advection')
ax[0, 1].legend(['Advection of bonds out of the $z$ domain',
                 'Advection of bonds out of the $\\theta$ domain'], loc='best')

ax[1, 1].plot(t, -force, t, -torque)
ax[1, 1].set_ylabel('Nondimensional force or torque')
ax[1, 1].set_title('Net Force and Torque')
ax[1, 1].legend(['Force', 'Torque'], loc='best')

for axs in np.ravel(ax):
    axs.grid(which='both')

plt.tight_layout()
plt.show()
