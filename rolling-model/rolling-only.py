from constructA import *
import matplotlib.pyplot as plt

use_torque = True

# Parameters
L = 2.0
M, N = 50, 50
z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                              np.linspace(-np.pi/2, np.pi/2, N+1),
                              indexing='ij')
z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')

h = z_mesh[1, 0] - z_mesh[0, 0]
nu = th_mesh[0, 1] - th_mesh[0, 0]
lam = nu/h

eta = .1
delta = 3
kap = 1

if use_torque:
    om_max, om_number = 200, 21
    eta_om = 1/5000

    omegas = np.linspace(0, om_max, om_number)
    torques = np.zeros(om_number)
    A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap)
    for i in range(om_number):
        torques[i] = find_torque_roll(A, B, C, D, R, omegas[i], omegas[i],
                                      lam, nu, h, kap, M, z_mesh, th_mesh)[0]

    plt.plot(omegas, torques)
    plt.xlabel('Rotation rate ($\omega$)')
    plt.ylabel('Torque ($\\tau$)')
    plt.show()

    om_f = omegas - torques/eta_om
    sorted_om_f = np.sort(om_f)

    plt.plot(om_f, omegas, 'b-')
    plt.plot(sorted_om_f, sorted_om_f, 'k--')
    plt.xlabel('Applied rotation rate ($\omega_f$)')
    plt.ylabel('Rotation rate ($\omega$)')
    plt.show()
else:
    v_max, v_number = 200, 201
    eta_f = 1/3000

    vees = np.linspace(0, v_max, v_number)
    forces = np.zeros(v_number)
    A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap)
    for i in range(v_number):
        forces[i] = find_force_roll(A, B, C, D, R, vees[i], vees[i],
                                      lam, nu, h, kap, M, z_mesh, th_mesh)[0]

    plt.plot(vees, forces)
    plt.xlabel('Rolling velocity ($v$)')
    plt.ylabel('Force ($f\'_h$)')
    plt.show()

    v_f = vees - forces/eta_f
    sorted_v_f = np.sort(v_f)

    plt.plot(v_f, vees, 'b-')
    plt.plot(sorted_v_f, sorted_v_f, 'k--')
    plt.xlabel('Applied rolling velocity ($v_f$)')
    plt.ylabel('Rolling velocity ($v$)')
    plt.show()
