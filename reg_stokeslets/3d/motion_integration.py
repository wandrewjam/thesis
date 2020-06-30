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


def evaluate_motion_equations(h, e_m, forces, torques, a=1.0, b=1.0,
                              n_nodes=8):
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

    dx1, dx2, dx3 = trans_vels
    dem1, dem2, dem3 = np.cross(rot_vels, e_m)

    return dx1, dx2, dx3, dem1, dem2, dem3


def time_step(dt, x1, x2, x3, e_m, forces, torques, n_nodes=8):
    dx1, dx2, dx3, dem1, dem2, dem3 = evaluate_motion_equations(
        x1, e_m, forces, torques, n_nodes=n_nodes)
    new_x1 = x1 + dt * dx1
    new_x2 = x2 + dt * dx2
    new_x3 = x3 + dt * dx3
    new_e_m = e_m + dt * np.array([dem1, dem2, dem3])
    new_e_m /= np.linalg.norm(new_e_m)

    return new_x1, new_x2, new_x3, new_e_m


def integrate_motion(t_span, num_steps, init, n_nodes):
    x1, x2, x3 = np.zeros(shape=(3, num_steps+1))
    e_m = np.zeros(shape=(3, num_steps+1))
    x1[0], x2[0], x3[0] = init[:3]
    e_m[:, 0] = init[3:]
    t_length = t_span[1] - t_span[0]
    dt = t_length / num_steps
    forces, torques = np.zeros((3, 1)), np.zeros((3, 1))

    for i in range(num_steps):
        x1[i+1], x2[i+1], x3[i+1], e_m[:, i+1] = (
            time_step(dt, x1[i], x2[i], x3[i], e_m[:, i], forces, torques, n_nodes=n_nodes)
        )

    return x1, x2, x3, e_m


def f(orient, t):
    theta, phi = orient
    dtheta = np.cos(theta) / (2 * np.tan(phi)) * .92368
    dphi = np.sin(theta) / 2 * .92368
    return np.array([dtheta, dphi])


def main():
    init = np.zeros(6)
    init[0] = 1.5431
    init[3] = np.cos(np.pi/8)
    init[4] = np.sin(np.pi/8)
    n_nodes = 8
    x1, x2, x3, e_m = integrate_motion([0., 10.], 200, init, n_nodes)
    t = np.linspace(0, 10, num=201)

    # Find analytic solution
    an_sol = odeint(f, [0, np.pi/8], t)
    e1_exact = np.cos(an_sol[:, 1])
    e2_exact = np.sin(an_sol[:, 1]) * np.cos(an_sol[:, 0])
    e3_exact = np.sin(an_sol[:, 1]) * np.sin(an_sol[:, 0])

    x3_exact = 1.5431 * .92185 * t
    plt.plot(t, x1, t, x2, t, x3, t, x3_exact)
    plt.legend(['x1', 'x2', 'x3', 'x3 exact'])
    plt.show()

    plt.plot(t, e_m[0], t, e_m[1], t, e_m[2],
             t, e1_exact, t, e2_exact, t, e3_exact)
    plt.legend(['$e_{m1}$ approx', '$e_{m2}$ approx', '$e_{m3}$ approx',
                '$e_{m1}$ exact', '$e_{m2}$ exact', '$e_{m3}$ exact'])
    plt.show()

    plt.plot(t, e_m[0] - e1_exact, t, e_m[1] - e2_exact, t, e_m[2] - e3_exact)
    plt.show()


if __name__ == '__main__':
    main()
