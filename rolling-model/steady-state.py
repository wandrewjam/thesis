from constructA import *
import matplotlib.pyplot as plt

# Parameters
L = 2.0
M, N = 100, 100
om_max, om_number = 300, 101
z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                              np.linspace(-np.pi/2, np.pi/2, N+1),
                              indexing='ij')
z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')

h = z_mesh[1, 0] - z_mesh[0, 0]
nu = th_mesh[0, 1] - th_mesh[0, 0]
lam = nu/h

d_prime = 0
eta = .1
delta = 3
kap = 1
radius = 1
xi = 1000

omegas = np.linspace(0, om_max, om_number)
torques = np.zeros(om_number)
A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime)
for i in range(om_number):
    torques[i] = find_torque_roll(A, B, C, D, R, omegas[i], (1+d_prime)*omegas[i],
                                  lam, nu, h, kap, M, z_mesh, th_mesh, d_prime)[0]

plt.plot(omegas, torques)
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
