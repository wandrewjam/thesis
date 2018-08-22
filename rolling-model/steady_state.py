from constructA import *
import matplotlib.pyplot as plt

# Parameters
L = 2.0
M, N = 50, 50
om_max, om_number = 300, 101
v_max, v_number = 200, 6
z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                              np.linspace(-np.pi/2, np.pi/2, N+1),
                              indexing='ij')
z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')

h = z_mesh[1, 0] - z_mesh[0, 0]
nu = th_mesh[0, 1] - th_mesh[0, 0]
lam = nu/h

d_prime = .1
eta = .1
delta = 3
kap = 1
radius = 1
xi = 1000

omegas = np.linspace(0, om_max, om_number)
vees = np.linspace(0, v_max, v_number)
torques, forces = np.zeros(shape=(om_number, v_number)), np.zeros(shape=(om_number, v_number))
A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime)
for i in range(om_number):
    for j in range(v_number):
        torques[i, j], forces[i, j] = \
            find_torque_roll(A, B, C, D, R, omegas[i], vees[j], lam, nu,
                             h, kap, M, z_mesh, th_mesh, d_prime)[:-1]

for j in range(v_number):
    plt.plot(omegas, torques[:,j])
plt.xlabel('Rotation rate ($\omega$)')
plt.ylabel('Torque ($\\tau$)')
plt.show()

om_f = omegas - torques*xi*20
sorted_om_f = np.sort(om_f)

plt.plot(om_f, omegas, 'bo-')
plt.plot(sorted_om_f, sorted_om_f, 'k--')
plt.xlabel('Applied rotation rate ($\omega_f$)')
plt.ylabel('Rotation rate ($\omega$)')
plt.show()
