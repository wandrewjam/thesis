import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from resistance_matrix_test import generate_resistance_matrices
from dist_convergence_test import spheroid_surface_area


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
        generate_resistance_matrices(eps, n_nodes, a=a, b=b, domain='wall',
                                     distance=h, theta=theta, phi=phi))

    res_matrix = np.block([[t_matrix, p_matrix], [pt_matrix, r_matrix]])
    gen_forces = np.block([[shear_f], [shear_t]])
    gen_vels = np.linalg.solve(res_matrix, -gen_forces)
    trans_vels, rot_vels = np.squeeze(gen_vels[:3]), np.squeeze(gen_vels[3:])
    # trans_vels = (np.linalg.solve(t_matrix, forces + shear_f)
    #               + np.linalg.solve(p_matrix, torques + shear_t))
    # rot_vels = (np.linalg.solve(pt_matrix, forces + shear_f)
    #             + np.linalg.solve(r_matrix, torques + shear_t))

    velocity_error = np.amax(np.abs(
        np.concatenate((trans_vels, rot_vels)) - exact_vels
    ))

    dx1, dx2, dx3 = trans_vels
    dem1, dem2, dem3 = np.cross(rot_vels, e_m)

    return dx1, dx2, dx3, dem1, dem2, dem3, velocity_error


def time_step(dt, x1, x2, x3, e_m, forces, torques, exact_vels, n_nodes=8):
    dx1, dx2, dx3, dem1, dem2, dem3, velocity_error = evaluate_motion_equations(x1, e_m, forces, torques, exact_vels, n_nodes=n_nodes)
    new_x1 = x1 + dt * dx1
    new_x2 = x2 + dt * dx2
    new_x3 = x3 + dt * dx3
    new_e_m = e_m + dt * np.array([dem1, dem2, dem3])
    new_e_m /= np.linalg.norm(new_e_m)

    return new_x1, new_x2, new_x3, new_e_m, velocity_error


def integrate_motion(t_span, num_steps, init, n_nodes, exact_vels):
    x1, x2, x3 = np.zeros(shape=(3, num_steps+1))
    e_m = np.zeros(shape=(3, num_steps+1))
    x1[0], x2[0], x3[0] = init[:3]
    e_m[:, 0] = init[3:]
    t_length = t_span[1] - t_span[0]
    dt = t_length / num_steps
    forces, torques = np.zeros((3, 1)), np.zeros((3, 1))
    errs = np.zeros(shape=num_steps+1)

    for i in range(num_steps):
        x1[i+1], x2[i+1], x3[i+1], e_m[:, i+1], errs[i+1] = (
            time_step(dt, x1[i], x2[i], x3[i], e_m[:, i], forces, torques,
                      exact_vels, n_nodes=n_nodes)
        )

    return x1, x2, x3, e_m, errs


def main():
    import os
    plot_dir = os.path.expanduser('~/thesis/meeting-notes/summer-20/'
                                  'notes_070120/')
    save_plots = True
    init = np.zeros(6)

    # distance = 1.005004
    # rot_correction = 0.50818
    # trn_correction = 0.47861
    # plot_num = 5

    # For N = 16
    # plot_num = 9

    # distance = 1.0453
    # rot_correction = .67462
    # trn_correction = .65375
    # plot_num = 4

    # For N = 16
    # plot_num = 8

    # distance = 1.1276
    # rot_correction = 0.77916
    # trn_correction = 0.76692
    # plot_num = 3

    # For N = 16
    # plot_num = 7

    distance = 1.5431
    rot_correction = 0.92368
    trn_correction = .92185
    plot_num = 2

    # For N = 16
    # plot_num = 6

    # distance = 0
    # rot_correction = 1.0
    # trn_correction = 1.0
    # plot_num = 1

    exact_trn_vel = trn_correction * distance * np.array([0, 0, 1])
    exact_rot_vel = rot_correction / 2 * np.array([0, -1, 0])
    exact_vel = np.concatenate((exact_trn_vel, exact_rot_vel))
    init[0] = distance
    init[3] = 1
    init[4] = 0
    n_nodes = 8
    x1, x2, x3, e_m, errs = integrate_motion(
        [0., 10.], 200, init, n_nodes, exact_vel)
    t = np.linspace(0, 10, num=201)

    # Find analytic solution
    t_adj = t * rot_correction / 2
    ex0, ey0, ez0 = init[3:]
    e1_exact = ex0 * np.cos(t_adj) - ez0 * np.sin(t_adj)
    e2_exact = ey0 * np.ones(shape=t_adj.shape)
    e3_exact = ez0 * np.cos(t_adj) + ex0 * np.sin(t_adj)
    x3_exact = distance * trn_correction * t

    # Plot numerical and analytical solutions
    fig1, ax1 = plt.subplots()
    ax1.plot(t, x1, t, x2, t, x3, t, x3_exact)
    ax1.legend(['x1', 'x2', 'x3', 'x3 exact'])
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
    ax4.plot(t, errs)
    ax4.set_xlabel('Time elapsed')
    ax4.set_ylabel('Velocity error')
    if save_plots:
        fig4.savefig(plot_dir + 'vel_err_plot{}'.format(plot_num),
                     bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
