import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

d_prime = 0.1
om_number, v_number = 51, 51
eta_om, eta_v = 0.0001, 0.0001
sat = True

data = np.load('./data/ss_sweep_dprime{:g}_num{:g}_sat{:b}.npz'.format(d_prime, om_number, sat))
om_f, v_f, omegas, vees, torques, forces = data['om_f'], data['v_f'], data['omegas'], \
                                           data['vees'], data['torques'], data['forces']

# Plotting code
fig, ax1 = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')

for j in range(v_number):
    ax1[0, 0].plot(omegas, torques[:, j])
    ax1[1, 0].plot(omegas, forces[:, j])
for j in range(om_number):
    ax1[0, 1].plot(vees, torques[j, :])
    ax1[1, 1].plot(vees, forces[j, :])
ax1[0, 0].set_ylabel('Torque ($\\tau$)')
ax1[1, 0].set_xlabel('Rotation rate ($\omega$)')
ax1[1, 0].set_ylabel('Force ($f_h$)')
ax1[1, 1].set_xlabel('Velocity ($v$)')

fig2 = plt.figure()
ax2 = list()
ax2.append(fig2.add_subplot(121, projection='3d'))
ax2.append(fig2.add_subplot(122, projection='3d'))

# Mask values where v or omega are negative
om_matrix = np.ma.array(np.broadcast_to(omegas[:, None], shape=(om_number, v_number)))
v_matrix = np.ma.array(np.broadcast_to(vees, shape=(om_number, v_number)))
om_matrix[(om_f < 0) | (v_f < 0)] = np.ma.masked
v_matrix[(om_f < 0) | (v_f < 0)] = np.ma.masked

ax2[0].scatter(om_f, v_f, om_matrix, c=om_matrix.flatten())
ax2[1].scatter(om_f, v_f, v_matrix, c=v_matrix.flatten())

# Set axis labels
ax2[0].set_xlabel('$\\omega_f$')
ax2[0].set_ylabel('$v_f$')
ax2[0].set_zlabel('$\\omega$')

ax2[1].set_xlabel('$\\omega_f$')
ax2[1].set_ylabel('$v_f$')
ax2[1].set_zlabel('$v$')


# for j in range(v_number):
#     ax2[0].plot(om_f[:, j], omegas)
# for j in range(om_number):
#     ax2[1].plot(v_f[j, :], vees)
# ax2[0].set_ylabel('Rotation rate ($\omega$)')
# ax2[0].set_xlabel('Applied rotation rate ($\omega_f$)')
# ax2[1].set_ylabel('Velocity ($v$)')
# ax2[1].set_xlabel('Applied velocity ($v_f$)')

plt.show()
print('Done')

# NOTE: I was working on the plotting code, test first to see what I need
