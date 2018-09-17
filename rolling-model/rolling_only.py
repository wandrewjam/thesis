from constructA import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def rolling_only_sweep(use_torque=True, L=2.5, M=100, N=100, eta=.1, delta=3, kap=1, eta_v=.01, eta_om=.01,
                 om_max=200, om_number=101, v_max=200, v_number=101, plot=True):

    # Mesh setups
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')
    z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')

    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    lam = nu/h

    if use_torque:
        omegas = np.linspace(0, om_max, om_number)
        torques = np.zeros(om_number)
        A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap)
        start = timer()
        for i in range(om_number):
            torques[i] = find_torque_roll(A, B, C, D, R, omegas[i], omegas[i],
                                          lam, nu, h, kap, M, z_mesh, th_mesh)[0]
            print('Completed {:d} of {:d} loops'.format(i+1, om_number))
        end = timer()

        print('A single loop took {:g} seconds on average'.format((end-start)/om_number))

        if plot:
            plt.plot(omegas, torques)
            plt.xlabel('Rotation rate ($\omega$)')
            plt.ylabel('Torque ($\\tau$)')
            plt.show()

        om_f = omegas - torques/eta_om
        sorted_om_f = np.sort(om_f)

        if plot:
            plt.plot(om_f, omegas, 'b-')
            plt.plot(sorted_om_f, sorted_om_f, 'k--')
            plt.xlabel('Applied rotation rate ($\omega_f$)')
            plt.ylabel('Rotation rate ($\omega$)')
            plt.show()

        return omegas, torques, om_f
    else:
        vees = np.linspace(0, v_max, v_number)
        forces = np.zeros(v_number)
        A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap)
        for i in range(v_number):
            forces[i] = find_force_roll(A, B, C, D, R, vees[i], vees[i],
                                        lam, nu, h, kap, M, z_mesh, th_mesh)[0]

        if plot:
            plt.plot(vees, forces)
            plt.xlabel('Rolling velocity ($v$)')
            plt.ylabel('Force ($f\'_h$)')
            plt.show()

        v_f = vees - forces/eta_v
        sorted_v_f = np.sort(v_f)

        if plot:
            plt.plot(v_f, vees, 'b-')
            plt.plot(sorted_v_f, sorted_v_f, 'k--')
            plt.xlabel('Applied rolling velocity ($v_f$)')
            plt.ylabel('Rolling velocity ($v$)')
            plt.show()

        return vees, forces, v_f


rolling_only_sweep(om_number=101, eta_om=.0001)
