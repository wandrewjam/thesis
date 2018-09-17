from time_dependent import time_dependent
import numpy as np
import matplotlib.pyplot as plt

v1, om1, m1, t = time_dependent(expt='unbound', gamma=0.0)

# gamma = 0*t**4/(t**4 + .02**4)
v2, om2, m2, _ = time_dependent(expt='unmoving', gamma=0.0)

fig1, ax1 = plt.subplots(ncols=2)
ax1[0].plot(t, v1)
ax1[0].plot(t, 22*np.ones(shape=t.shape), 'k--')
ax1[0].set_ylabel('ND linear velocity ($v$)')
ax1[0].set_title('Linear velocity of unbound platelet')
ax1[1].plot(t, om1)
ax1[1].plot(t, 20*np.ones(shape=t.shape), 'k--')
ax1[1].set_ylabel('ND angular velocity ($\\omega$)')
ax1[1].set_title('Angular velocity of unbound platelet')
for ax in ax1:
    ax.set_xlabel('Nondimensional time ($s$)')

fig2, ax2 = plt.subplots(ncols=2)

ax2[0].plot(t, v2)
ax2[0].plot(t, np.zeros_like(t), 'k--')
ax2[0].set_ylabel('ND linear velocity ($v$)')
ax2[0].set_title('Linear velocity of stably bound platelet')
ax2[1].plot(t, om2)
ax2[1].plot(t, np.zeros_like(t), 'k--')
ax2[1].set_ylabel('ND angular velocity ($\\omega$)')
ax2[1].set_title('Angular velocity of stably bound platelet')
for ax in ax2:
    ax.set_xlabel('Nondimensional time ($s$)')
plt.show()
