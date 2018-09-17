from constructA import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer


def steady_state_sweep(L=2.5, M=100, N=100, d_prime=.1, eta=.1, delta=3, kap=1, eta_v=.01, eta_om=.01,
                       om_max=200, om_number=21, v_max=200, v_number=21, saturation=True, plot=True):
    start = timer()
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')
    z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')

    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    lam = nu/h

    A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime, saturation)

    if d_prime > 0:
        omegas = np.linspace(0, om_max, om_number)
        torques = np.zeros(shape=(om_number, v_number))
        vees = np.linspace(0, v_max, v_number)
        forces = np.zeros(shape=(om_number, v_number))

        for i in range(om_number):
            for j in range(v_number):
                torques[i, j], forces[i, j] = \
                    find_forces(A, B, C, D, R, omegas[i], vees[j], lam, nu,
                                h, kap, M, z_mesh, th_mesh, d_prime)[:-1]
                print('Completed {:d} of {:d} inner loops'.format(j+1, v_number))
            print('Completed {:d} of {:d} outer loops'.format(i+1, om_number))

        v_f, om_f = vees - forces/eta_v, omegas[:, None] - torques/eta_om  # Note vees and omegas are broadcasted here
        end = timer()

        print('This sweep took {:g} seconds total, and {:g} seconds for each solve'.format(end-start,
                                                                                           (end-start)/(om_number*v_number)))

        if plot:
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')

            for j in range(v_number):
                ax[0, 0].plot(omegas, torques[:, j])
                ax[1, 0].plot(omegas, forces[:, j])
            for j in range(om_number):
                ax[0, 1].plot(vees, torques[j, :])
                ax[1, 1].plot(vees, forces[j, :])
            ax[0, 0].set_ylabel('Torque ($\\tau$)')
            ax[1, 0].set_xlabel('Rotation rate ($\omega$)')
            ax[1, 0].set_ylabel('Force ($f_h$)')
            ax[1, 1].set_xlabel('Velocity ($v$)')

            fig2 = plt.figure()
            ax2 = list()
            ax2.append(fig2.add_subplot(121, projection='3d'))
            ax2.append(fig2.add_subplot(122, projection='3d'))

            ax2[0].scatter(om_f, v_f, np.broadcast_to(omegas[:, None], shape=(om_number, v_number)))
            ax2[1].scatter(om_f, v_f, np.broadcast_to(vees, shape=(om_number, v_number)))


            # for j in range(v_number):
            #     ax2[0].plot(om_f[:, j], omegas)
            # for j in range(om_number):
            #     ax2[1].plot(v_f[j, :], vees)
            # ax2[0].set_ylabel('Rotation rate ($\omega$)')
            # ax2[0].set_xlabel('Applied rotation rate ($\omega_f$)')
            # ax2[1].set_ylabel('Velocity ($v$)')
            # ax2[1].set_xlabel('Applied velocity ($v_f$)')

            plt.show()

        return v_f, om_f, vees, omegas, forces, torques
    elif d_prime == 0:
        omegas = np.linspace(0, om_max, om_number)
        torques = np.zeros(shape=om_number)

        for i in range(om_number):
            torques[i] = find_torque_roll(A, B, C, D, R, omegas[i], omegas[i],
                                          lam, nu, h, kap, M, z_mesh, th_mesh)[0]
            print('Completed {:d} of {:d} loops'.format(i+1, om_number))

        om_f = omegas - torques/eta_om
        end = timer()

        print('This sweep took {:g} seconds total, and {:g} seconds for each solve'.format(end-start, (end-start)/om_number))
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=2)

            ax[0].plot(omegas, torques)
            ax[0].set_xlabel('Rotation rate ($\omega$)')
            ax[0].set_ylabel('Torque ($\\tau$)')

            sorted_om_f = np.sort(om_f)
            ax[1].plot(om_f, omegas)
            ax[1].plot(sorted_om_f, sorted_om_f, 'k--')
            ax[1].set_xlabel('Applied rotation rate ($\omega_f$)')
            ax[1].set_ylabel('Rotation rate ($\omega$)')

            plt.show()

        return om_f, omegas, torques


# steady_state_sweep(d_prime=0.1, om_number=11, eta_om=.0001, v_number=11, eta_v=.0001, saturation=False)
