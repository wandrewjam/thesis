import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from resistance_matrix_test import generate_resistance_matrices
from dist_convergence_test import spheroid_surface_area
from timeit import default_timer as timer


def eps_picker(n_nodes, a, b):
    c, n = 0.6, 1.0
    surf_area = spheroid_surface_area(a, b)
    h = np.sqrt(surf_area / (6 * n_nodes ** 2 + 2))
    eps = c * h**n
    return eps


def evaluate_motion_equations(h, e_m, forces, torques, exact_vels, a=1.0,
                              b=1.0, n_nodes=8):
    eps = eps_picker(n_nodes, a, b)
    theta = np.arctan2(e_m[2], e_m[1])
    phi = np.arccos(e_m[0])
    t_matrix, p_matrix, pt_matrix, r_matrix, shear_f, shear_t = (
        generate_resistance_matrices(eps, n_nodes, a=a, b=b, domain='free',
                                     distance=h, theta=theta, phi=phi))

    res_matrix = np.block([[t_matrix, p_matrix], [pt_matrix, r_matrix]])
    gen_forces = np.block([[shear_f], [shear_t]])
    gen_vels = np.linalg.solve(res_matrix, -gen_forces)
    trans_vels, rot_vels = np.squeeze(gen_vels[:3]), np.squeeze(gen_vels[3:])
    # trans_vels = (np.linalg.solve(t_matrix, forces + shear_f)
    #               + np.linalg.solve(p_matrix, torques + shear_t))
    # rot_vels = (np.linalg.solve(pt_matrix, forces + shear_f)
    #             + np.linalg.solve(r_matrix, torques + shear_t))

    exact = exact_vels(e_m)
    velocity_errors = np.concatenate((trans_vels, rot_vels)) - exact

    dx1, dx2, dx3 = trans_vels
    dem1, dem2, dem3 = np.cross(rot_vels, e_m)

    return dx1, dx2, dx3, dem1, dem2, dem3, velocity_errors


def time_step(dt, x1, x2, x3, e_m, forces, torques, exact_vels, n_nodes=8,
              a=1.0, b=1.0):
    dx1, dx2, dx3, dem1, dem2, dem3, velocity_errors = (
        evaluate_motion_equations(x1, e_m, forces, torques, exact_vels,
                                  n_nodes=n_nodes, a=a, b=b)
    )
    new_x1 = x1 + dt * dx1
    new_x2 = x2 + dt * dx2
    new_x3 = x3 + dt * dx3
    new_e_m = e_m + dt * np.array([dem1, dem2, dem3])
    new_e_m /= np.linalg.norm(new_e_m)

    return new_x1, new_x2, new_x3, new_e_m, velocity_errors


def integrate_motion(t_span, num_steps, init, n_nodes, exact_vels, a=1.0,
                     b=1.0):
    x1, x2, x3 = np.zeros(shape=(3, num_steps+1))
    e_m = np.zeros(shape=(3, num_steps+1))
    x1[0], x2[0], x3[0] = init[:3]
    e_m[:, 0] = init[3:]
    t_length = t_span[1] - t_span[0]
    dt = t_length / num_steps
    forces, torques = np.zeros((3, 1)), np.zeros((3, 1))
    errs = np.zeros(shape=(6, num_steps+1))

    for i in range(num_steps):
        x1[i+1], x2[i+1], x3[i+1], e_m[:, i+1], errs[:, i+1] = (
            time_step(dt, x1[i], x2[i], x3[i], e_m[:, i], forces, torques,
                      exact_vels, n_nodes=n_nodes, a=a, b=b)
        )

    return x1, x2, x3, e_m, errs


def main(plot_num):
    import os
    plot_dir = os.path.expanduser('~/Documents/thesis/meeting-notes/summer-20/'
                                  'notes_070120/')
    save_plots = False
    init = np.zeros(6)
    t_steps = 200

    # Set platelet geometry
    if 1 <= plot_num <= 9:
        a, b = 1.0, 1.0
    elif 11 <= plot_num:
        a, b = 1.5, 0.5
    else:
        raise ValueError('plot_num is invalid')

    # Set number of nodes
    if 1 <= plot_num <= 5 or 11 <= plot_num <= 15:
        n_nodes = 8
    elif 6 <= plot_num <= 9 or 16 == plot_num:
        n_nodes = 16
    else:
        raise ValueError('plot_num is invalid')

    # Set distance to wall
    if plot_num == 1:
        distance = 0.
        rot_correction = 1.0
        trn_correction = 1.0
    elif plot_num == 2 or plot_num == 6:
        distance = 1.5431
        rot_correction = 0.92368
        trn_correction = 0.92185
    elif plot_num == 3 or plot_num == 7:
        distance = 1.1276
        rot_correction = 0.77916
        trn_correction = 0.76692
    elif plot_num == 4 or plot_num == 8:
        distance = 1.0453
        rot_correction = 0.67462
        trn_correction = 0.65375
    elif plot_num == 5 or plot_num == 9:
        distance = 1.005004
        rot_correction = 0.50818
        trn_correction = 0.47861
    elif plot_num == 11 or plot_num == 16:
        distance = 0.
    elif plot_num == 12 or plot_num == 17:
        distance = 1.5
    elif plot_num == 13 or plot_num == 18:
        distance = 1.2
    elif plot_num == 14 or plot_num == 19:
        distance = 1.0
    else:
        raise ValueError('plot_num is invalid')

    # Set initial position and orientation
    init[0] = distance
    init[3] = 1
    init[4] = 0
    init[5] = 0

    # Find analytic solution
    t = np.linspace(0, 10, num=t_steps + 1)

    if 1 <= plot_num <= 9:
        def exact_vels(em):
            exact_trn_vel = trn_correction * distance * np.array([0, 0, 1])
            exact_rot_vel = rot_correction / 2 * np.array([0, -1, 0])
            vels = np.concatenate((exact_trn_vel, exact_rot_vel))
            return vels
        t_adj = t * rot_correction / 2

        ex0, ey0, ez0 = init[3:]
        e1_exact = ex0 * np.cos(t_adj) - ez0 * np.sin(t_adj)
        e2_exact = ey0 * np.ones(shape=t_adj.shape)
        e3_exact = ez0 * np.cos(t_adj) + ex0 * np.sin(t_adj)
        x3_exact = distance * trn_correction * t
    elif 11 == plot_num or 16 == plot_num:
        e = np.sqrt(1 - b**2 / a**2)
        xc = 2. / 3 * e**3 * (np.arctan(e / np.sqrt(1 - e**2))
                              - e * np.sqrt(1 - e**2)) ** (-1)
        yc = 2. / 3 * e**3 * (2 - e**2) * (
                e * np.sqrt(1 - e**2) - (1 - 2*e**2)
                * np.arctan(e / np.sqrt(1 - e**2))
        ) ** (-1)
        yh = 2. / 3 * e**5 * (
                e * np.sqrt(1 - e**2) - (1 - 2 * e**2)
                * np.arctan(e / np.sqrt(1 - e**2))
        ) ** (-1)
        eps = np.zeros(shape=(3, 3, 3))
        eps[0, 1, 2], eps[1, 2, 0], eps[2, 0, 1] = 1, 1, 1
        eps[0, 2, 1], eps[2, 1, 0], eps[1, 0, 2] = -1, -1, -1
        ros = np.zeros(shape=(3, 3))
        ros[0, 2], ros[2, 0] = 0.5, 0.5
        om_inf = np.array([0, -.5, 0])

        def exact_vels(em):
            trn_vels = np.zeros(3)
            I = np.eye(3)
            outer = em[:, None] * em[None, :]
            A = xc * outer + yc * (I - outer)
            rhs = yh / 2 * np.tensordot(np.dot(
                (eps[:, :, None, :] * em[None, None, :, None]
                 + eps[:, None, :, :] * em[None, :, None, None]), em
            ), ros) + np.dot(A, om_inf)
            vels = np.linalg.solve(A, rhs)
            return np.concatenate((trn_vels, vels))

        em_exact = np.zeros(shape=(3, t_steps + 1))
        em_exact[:, 0] = init[3:]
        dt = t[1] - t[0]
        for (i, t_el) in enumerate(t[1:]):
            d_em = exact_vels(em_exact[:, i])[3:]
            em_exact[:, i+1] = (em_exact[:, i]
                                + dt * np.cross(d_em, em_exact[:, i]))
            em_exact[:, i+1] /= np.linalg.norm(em_exact[:, i+1])

        e1_exact, e2_exact, e3_exact = em_exact
        x3_exact = np.zeros(t_steps+1)
    else:
        exact_nodes = 24

        def exact_vels(em):
            return np.zeros(6)

        ex_start = timer()
        x1_exact, x2_exact, x3_exact, em_exact = integrate_motion(
            [0., 10.], t_steps, init, exact_nodes,
            exact_vels, a=a, b=b)[:-1]
        ex_end = timer()

        e1_exact, e2_exact, e3_exact = em_exact

    # Integrate platelet motion
    start = timer()
    x1, x2, x3, e_m, errs = integrate_motion([0., 10.], t_steps, init,
                                             n_nodes, exact_vels, a=a, b=b)
    end = timer()

    try:
        print('Exact solve took {} seconds'.format(ex_end - ex_start))
    except NameError:
        pass

    print('Approx solve took {} seconds'.format(end - start))

    # Plot numerical and analytical solutions
    fig1, ax1 = plt.subplots()
    ax1.plot(t, x1, t, x2, t, x3, t, x3_exact)
    ax1.legend(['$x$', '$y$', '$z$', '$z$ exact'])
    ax1.set_xlabel('Time elapsed')
    ax1.set_ylabel('Center of mass position')
    if save_plots:
        fig1.savefig(plot_dir + 'com_plot{}'.format(plot_num),
                     bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(t, e_m[0], t, e_m[1], t, e_m[2],
             t, e1_exact, t, e2_exact, t, e3_exact)
    ax2.legend(['$e_{mx}$ approx', '$e_{my}$ approx', '$e_{mz}$ approx',
                '$e_{mx}$ exact', '$e_{my}$ exact', '$e_{mz}$ exact'])
    ax2.set_xlabel('Time elapsed')
    ax2.set_ylabel('Orientation components')
    if save_plots:
        fig2.savefig(plot_dir + 'orient_plot{}'.format(plot_num),
                     bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    fig3, ax3 = plt.subplots()
    ax3.plot(t, e_m[0] - e1_exact, t, e_m[1] - e2_exact, t, e_m[2] - e3_exact)
    ax3.legend(['$e_{mx}$ error', '$e_{my}$ error', '$e_{mz}$ error'])
    ax3.set_xlabel('Time elapsed')
    ax3.set_ylabel('Absolute error (approx - exact)')
    if save_plots:
        fig3.savefig(plot_dir + 'orient_err_plot{}'.format(plot_num),
                     bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    fig4, ax4 = plt.subplots()
    ax4.plot(t[1:], errs[:, 1:].T)
    ax4.set_xlabel('Time elapsed')
    ax4.set_ylabel('Velocity error (approx - exact)')
    ax4.legend(['$v_x$ error', '$v_y$ error', '$v_z$ error',
                '$\\omega_x$ error', '$\\omega_y$ error', '$\\omega_z$ error'])
    if save_plots:
        fig4.savefig(plot_dir + 'vel_err_plot{}'.format(plot_num),
                     bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    import sys

    expt = int(sys.argv[1])
    main(expt)
