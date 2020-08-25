import numpy as np
from resistance_matrix_test import generate_resistance_matrices
from dist_convergence_test import spheroid_surface_area
from sphere_integration_utils import generate_grid
from timeit import default_timer as timer


def read_parameter_file(filename):
    txt_dir = 'par-files/experiments/'
    parlist = [('filename', filename)]

    with open(txt_dir + filename + '.txt') as f:
        # Need the following keys: distance, ex0, ey0, ez0, t_steps,
        # stop, adaptive, a, b, n_nodes, order, domain
        while True:
            command = f.readline().split()
            if len(command) < 1:
                continue
            if command[0] == 'done':
                break

            key, value = command
            if key == 'num_expt':
                parlist.append((key, int(value)))
            else:
                parlist.append((key, float(value)))
    return dict(parlist)


def eps_picker(n_nodes, a, b):
    c, n = 0.6, 1.0
    surf_area = spheroid_surface_area(a, b)
    h = np.sqrt(surf_area / (6 * n_nodes ** 2 + 2))
    eps = c * h**n
    return eps


def n_picker(sep):
    c_star = 1.
    n_min, n_max = 16, 24
    s = spheroid_surface_area(a=1.5, b=.5)
    k = int(np.ceil(0.5 * np.sqrt((s * (c_star * sep / 0.6) ** (-2) - 2) / 6)))
    n_nodes = np.max([np.min([2 * k, n_max]), n_min])
    return n_nodes


def valid_orientation(n_nodes, e_m, distance, a, b):
    xi_mesh, eta_mesh, nodes, ind_map = generate_grid(n_nodes, a=a, b=b)
    theta = np.arctan2(e_m[2], e_m[1])
    phi = np.arccos(e_m[0])
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    rot_matrix = np.array([[cp, -sp, 0],
                           [ct * sp, ct * cp, -st],
                           [st * sp, st * cp, ct]])
    nodes = np.dot(rot_matrix, nodes.T).T
    nodes[:, 0] += distance
    return np.all(nodes[:, 0] > 0)


def find_min_separation(com_dist, e_m):
    mesh = generate_grid(n_nodes=36, a=1.5, b=0.5)[2]
    theta = np.arctan2(e_m[2], e_m[1])
    phi = np.arccos(e_m[0])
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    rot_matrix = np.array([[cp, -sp, 0],
                           [ct * sp, ct * cp, -st],
                           [st * sp, st * cp, ct]])
    mesh = np.dot(rot_matrix, mesh.T).T
    mesh[:, 0] += com_dist
    sep = np.amin(mesh[:, 0])
    return sep


def evaluate_motion_equations(h, e_m, forces, torques, exact_vels, a=1.0,
                              b=1.0, n_nodes=8, domain='free'):
    eps = eps_picker(n_nodes, a, b)
    theta = np.arctan2(e_m[2], e_m[1])
    phi = np.arccos(e_m[0])
    t_matrix, p_matrix, pt_matrix, r_matrix, shear_f, shear_t = (
        generate_resistance_matrices(eps, n_nodes, a=a, b=b, domain=domain,
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

    # Define a counter for the number of RHS evaluations
    evaluate_motion_equations.counter += 1

    return dx1, dx2, dx3, dem1, dem2, dem3, velocity_errors


def time_step(dt, x1, x2, x3, e_m, forces, torques, exact_vels, n_nodes=8,
              a=1.0, b=1.0, domain='free', order='2nd'):
    if order == '1st':
        dx1, dx2, dx3, dem1, dem2, dem3, velocity_errors = (
            evaluate_motion_equations(x1, e_m, forces, torques, exact_vels,
                                      a=a, b=b, n_nodes=n_nodes, domain=domain)
        )
        new_x1 = x1 + dt * dx1
        new_x2 = x2 + dt * dx2
        new_x3 = x3 + dt * dx3
        new_e_m = e_m + dt * np.array([dem1, dem2, dem3])
        new_e_m /= np.linalg.norm(new_e_m)

        # Check that we have a valid orientation
        if (not valid_orientation(n_nodes, new_e_m, distance=new_x1, a=a, b=b)
                and domain == 'wall'):
            # Then take 2 half time-steps
            tmp_x1, tmp_x2, tmp_x3, tmp_e_m = time_step(dt / 2, x1, x2, x3, e_m, forces, torques, exact_vels, n_nodes,
                                                        a, b, domain, order)[:-1]
            new_x1, new_x2, new_x3, new_e_m, velocity_errors = time_step(dt / 2, tmp_x1, tmp_x2, tmp_x3, tmp_e_m,
                                                                         forces, torques, exact_vels, n_nodes, a, b,
                                                                         domain, order)
    elif order == '2nd':
        dx1, dx2, dx3, dem1, dem2, dem3, velocity_errors = (
            evaluate_motion_equations(x1, e_m, forces, torques, exact_vels,
                                      a=a, b=b, n_nodes=n_nodes, domain=domain)
        )
        prd_x1 = x1 + dt * dx1
        prd_x2 = x2 + dt * dx2
        prd_x3 = x3 + dt * dx3
        prd_e_m = e_m + dt * np.array([dem1, dem2, dem3])
        prd_e_m /= np.linalg.norm(prd_e_m)

        try:
            dx1_p, dx2_p, dx3_p, dem1_p, dem2_p, dem3_p = (
                evaluate_motion_equations(prd_x1, prd_e_m, forces, torques,
                                          exact_vels, a=a, b=b,
                                          n_nodes=n_nodes, domain=domain)
            )[:-1]

            new_x1 = x1 + dt / 2 * (dx1 + dx1_p)
            new_x2 = x2 + dt / 2 * (dx2 + dx2_p)
            new_x3 = x3 + dt / 2 * (dx3 + dx3_p)
            new_e_m = e_m + dt / 2 * (np.array([dem1 + dem1_p, dem2 + dem2_p,
                                                dem3 + dem3_p]))
            new_e_m /= np.linalg.norm(new_e_m)

            if not valid_orientation(n_nodes, new_e_m, distance=new_x1,
                                     a=a, b=b) and domain == 'wall':
                raise AssertionError('next step will not be valid')
        except AssertionError:
            # If we get an invalid orientation, then take 2 half-steps
            tmp_x1, tmp_x2, tmp_x3, tmp_e_m = time_step(dt / 2, x1, x2, x3, e_m, forces, torques, exact_vels, n_nodes,
                                                        a, b, domain, order)[:-1]
            new_x1, new_x2, new_x3, new_e_m, velocity_errors = time_step(dt / 2, tmp_x1, tmp_x2, tmp_x3, tmp_e_m,
                                                                         forces, torques, exact_vels, n_nodes, a, b,
                                                                         domain, order)
    else:
        raise ValueError('order is not valid')

    return new_x1, new_x2, new_x3, new_e_m, velocity_errors


def integrate_motion(t_span, num_steps, init, exact_vels, n_nodes=None, a=1.0,
                     b=1.0, domain='free', order='2nd', adaptive=True):
    # Check that we have a valid combination of n_nodes and adaptive
    assert n_nodes > 0 or adaptive

    x1, x2, x3 = np.zeros(shape=(3, num_steps+1))
    e_m = np.zeros(shape=(3, num_steps+1))
    x1[0], x2[0], x3[0] = init[:3]
    e_m[:, 0] = init[3:]
    t_length = t_span[1] - t_span[0]
    dt = t_length / num_steps
    forces, torques = np.zeros((3, 1)), np.zeros((3, 1))
    errs = np.zeros(shape=(6, num_steps+1))
    node_array, sep_array = np.zeros(shape=(2, num_steps+1))

    try:
        for i in range(num_steps):
            if adaptive:
                # Find the wall separation
                sep = find_min_separation(x1[i], e_m[:,i])
                sep_array[i] = sep

                # Pick n_nodes based on separation
                n_nodes = n_picker(sep)
            node_array[i] = n_nodes
            x1[i+1], x2[i+1], x3[i+1], e_m[:, i+1], errs[:, i+1] = (
                time_step(dt, x1[i], x2[i], x3[i], e_m[:, i], forces, torques, exact_vels, n_nodes=n_nodes, a=a, b=b,
                          domain=domain, order=order)
            )
    except AssertionError:
        print('Encountered an assertion error while integrating. Halting and '
              'outputting computation results so far.')
        pass

    return x1, x2, x3, e_m, errs, node_array, sep_array


def main(plot_num, server='mac'):
    data_dir = 'data/'
    init = np.zeros(6)

    if server == 'mac':
        stop = 10.
        t_steps = 50
    elif server == 'linux':
        stop = 50.
        t_steps = 400
    else:
        raise ValueError('server is an invalid value')
    order = '2nd'

    # Initialize the function counter
    evaluate_motion_equations.counter = 0

    # Set platelet geometry
    if 1 <= plot_num <= 9:
        a, b = 1.0, 1.0
    elif 11 <= plot_num:
        a, b = 1.5, 0.5
    else:
        raise ValueError('plot_num is invalid')

    # Set number of nodes
    if (1 <= plot_num <= 5 or 11 <= plot_num <= 15 or 21 <= plot_num <= 25
            or 31 == plot_num or 41 <= plot_num <= 45 or 51 <= plot_num <= 55
            or 71 <= plot_num <= 74 or 81 <= plot_num <= 84
            or 91 <= plot_num <= 94):
        n_nodes = 8
    elif (6 <= plot_num <= 9 or 16 <= plot_num <= 19 or 26 <= plot_num <= 29
          or 36 == plot_num or 46 <= plot_num <= 49 or 56 <= plot_num <= 59
          or 76 <= plot_num <= 79 or 86 <= plot_num <= 89
          or 96 <= plot_num <= 99):
        n_nodes = 16
    else:
        raise ValueError('plot_num is invalid')

    # Set distance to wall
    if plot_num == 1:
        distance = 0.
        ex0, ey0, ez0 = 1., 0., 0.
        rot_correction = 1.0
        trn_correction = 1.0
        adaptive = False
    elif plot_num == 2 or plot_num == 6:
        distance = 1.5431
        ex0, ey0, ez0 = 1., 0., 0.
        rot_correction = 0.92368
        trn_correction = 0.92185
        adaptive = False
    elif plot_num == 3 or plot_num == 7:
        distance = 1.1276
        ex0, ey0, ez0 = 1., 0., 0.
        rot_correction = 0.77916
        trn_correction = 0.76692
        adaptive = False
    elif plot_num == 4 or plot_num == 8:
        distance = 1.0453
        ex0, ey0, ez0 = 1., 0., 0.
        rot_correction = 0.67462
        trn_correction = 0.65375
        adaptive = False
    elif plot_num == 5 or plot_num == 9:
        distance = 1.005004
        ex0, ey0, ez0 = 1., 0., 0.
        rot_correction = 0.50818
        trn_correction = 0.47861
        adaptive = False
    elif plot_num == 11 or plot_num == 16:
        distance = 0.
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = False
    elif plot_num == 12 or plot_num == 17:
        distance = 1.5
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = False
    elif plot_num == 13 or plot_num == 18:
        distance = 1.2
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = False
    elif plot_num == 14 or plot_num == 19:
        distance = 1.0
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = False
    elif plot_num == 21 or plot_num == 26:
        distance = 0.
        ex0, ey0, ez0 = np.sqrt(2)/2, np.sqrt(2)/2, 0.
        adaptive = False
    elif plot_num == 22 or plot_num == 27:
        distance = 1.5
        ex0, ey0, ez0 = np.sqrt(2)/2, np.sqrt(2)/2, 0.
        adaptive = False
    elif plot_num == 23 or plot_num == 28:
        distance = 1.2
        ex0, ey0, ez0 = np.sqrt(2)/2, np.sqrt(2)/2, 0.
        adaptive = False
    elif plot_num == 24 or plot_num == 29:
        distance = 1.0
        ex0, ey0, ez0 = np.sqrt(2)/2, np.sqrt(2)/2, 0.
        adaptive = False
    elif plot_num == 31 or plot_num == 36:
        distance = 0.
        ex0, ey0, ez0 = 0, 1., 0
        adaptive = False
    elif plot_num == 41 or plot_num == 46:
        distance = 0.
        ex0, ey0, ez0 = np.cos(np.pi / 8), np.sin(np.pi / 8), 0.
        adaptive = False
    elif plot_num == 42 or plot_num == 47:
        distance = 1.5
        ex0, ey0, ez0 = np.cos(np.pi / 8), np.sin(np.pi / 8), 0.
        adaptive = False
    elif plot_num == 43 or plot_num == 48:
        distance = 1.2
        ex0, ey0, ez0 = np.cos(np.pi / 8), np.sin(np.pi / 8), 0.
        adaptive = False
    elif plot_num == 44 or plot_num == 49:
        distance = 1.0
        ex0, ey0, ez0 = np.cos(np.pi / 8), np.sin(np.pi / 8), 0.
        adaptive = False
    elif plot_num == 51 or plot_num == 56:
        distance = 0.8
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = False
    elif plot_num == 52 or plot_num == 57:
        distance = 0.6
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = False
    elif plot_num == 71:
        distance = 1.5
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = True
    elif plot_num == 72:
        distance = 1.2
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = True
    elif plot_num == 73:
        distance = 1.0
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = True
    elif plot_num == 74:
        distance = 0.8
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = True
    elif plot_num == 76:
        distance = 1.1
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = True
    elif plot_num == 77:
        distance = 0.9
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = True
    elif plot_num == 81 or plot_num == 86:
        distance = 0.
        ex0, ey0, ez0 = 1., 0., 0.
        adaptive = False
    elif plot_num == 91 or plot_num == 96:
        distance = 0.
        ex0, ey0, ez0 = np.sqrt(2)/2, np.sqrt(2)/2, 0.
        adaptive = False
    else:
        raise ValueError('plot_num is invalid')

    if adaptive:
        stop *= 2
        t_steps *= 2

    # Set initial position and orientation
    init[0] = distance
    init[3] = ex0
    init[4] = ey0
    init[5] = ez0

    # Find analytic solution
    t = np.linspace(0, stop, num=t_steps + 1)

    if 1 <= plot_num <= 9:
        def exact_vels(em):
            exact_trn_vel = trn_correction * distance * np.array([0, 0, 1])
            exact_rot_vel = rot_correction / 2 * np.array([0, -1, 0])
            vels = np.concatenate((exact_trn_vel, exact_rot_vel))
            return vels

        t_adj = t * rot_correction / 2

        ex0, ey0, ez0 = init[3:]
        e1_fine = ex0 * np.cos(t_adj) - ez0 * np.sin(t_adj)
        e2_fine = ey0 * np.ones(shape=t_adj.shape)
        e3_fine = ez0 * np.cos(t_adj) + ex0 * np.sin(t_adj)

        x1_fine, x2_fine = np.zeros((2, t_steps+1))
        x3_fine = distance * trn_correction * t

        np.savez(data_dir + 'fine' + str(plot_num), x1_fine, x2_fine,
                 x3_fine, e1_fine, e2_fine, e3_fine, t, x1=x1_fine,
                 x2=x2_fine, x3=x3_fine, e1=e1_fine, e2=e2_fine,
                 e3=e3_fine, t=t)

        exact_solution = True

    elif (11 == plot_num or 16 == plot_num or 21 == plot_num or 26 == plot_num
          or 31 == plot_num or 36 == plot_num or 41 == plot_num
          or 46 == plot_num):
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

        em_fine = np.zeros(shape=(3, t_steps + 1))
        em_fine[:, 0] = init[3:]
        dt = t[1] - t[0]
        for (i, t_el) in enumerate(t[1:]):
            d_em = exact_vels(em_fine[:, i])[3:]
            cur = np.cross(d_em, em_fine[:, i])
            tmp = (em_fine[:, i] + dt * cur)
            tmp /= np.linalg.norm(tmp)
            prd = np.cross(exact_vels(tmp)[3:], tmp)
            em_fine[:, i+1] = (em_fine[:, i] + dt / 2 * (cur + prd))
            em_fine[:, i+1] /= np.linalg.norm(em_fine[:, i+1])

        e1_fine, e2_fine, e3_fine = em_fine
        x1_fine, x2_fine = np.zeros((2, t_steps+1))
        x3_fine = np.zeros(t_steps+1)
        np.savez(data_dir + 'fine' + str(plot_num), x1_fine, x2_fine,
                 x3_fine, e1_fine, e2_fine, e3_fine, t, x1=x1_fine,
                 x2=x2_fine, x3=x3_fine, e1=e1_fine, e2=e2_fine,
                 e3=e3_fine, t=t)

        exact_solution = True
    else:
        if server == 'mac':
            fine_nodes = 8
        elif server == 'linux':
            fine_nodes = 24
        else:
            raise ValueError('server is an invalid value')

        def exact_vels(em):
            return np.zeros(6)

        # Initialize counter and timer for fine simulation
        evaluate_motion_equations.counter = 0
        fine_start = timer()

        # Run the fine simulations
        if plot_num < 80:
            fine_result = integrate_motion(
                [0., stop], t_steps, init, exact_vels, fine_nodes, a=a,
                b=b, domain='wall', order=order, adaptive=False)
        else:
            fine_result = integrate_motion(
                [0., stop], t_steps, init, exact_vels, fine_nodes, a=a,
                b=b, domain='free', order=order, adaptive=False)
        x1_fine, x2_fine, x3_fine, em_fine = fine_result[:4]

        # Save the end state after the fine simulations
        fine_end = timer()
        fine_counter = evaluate_motion_equations.counter

        e1_fine, e2_fine, e3_fine = em_fine

        np.savez(data_dir + 'fine' + str(plot_num), x1_fine, x2_fine,
                 x3_fine, e1_fine, e2_fine, e3_fine, t, x1=x1_fine,
                 x2=x2_fine, x3=x3_fine, e1=e1_fine, e2=e2_fine,
                 e3=e3_fine, t=t)

        exact_solution = False

    if distance == 0:
        domain = 'free'
    elif distance > 0:
        domain = 'wall'
    else:
        raise ValueError('value of distance is unexpected')

    # Initialize counter and timer for coarse simulation
    start = timer()
    evaluate_motion_equations.counter = 0

    # Run the coarse simulations
    coarse_result = integrate_motion(
        [0., stop], t_steps, init, exact_vels, n_nodes, a=a, b=b,
        domain=domain, order=order, adaptive=adaptive)
    x1, x2, x3, e_m, errs, node_array, sep_array = coarse_result

    # Save the end state after the coarse simulations
    end = timer()
    coarse_counter = evaluate_motion_equations.counter

    e1, e2, e3 = e_m
    np.savez(data_dir + 'coarse' + str(plot_num), x1, x2, x3, e1, e2, e3, t,
             errs, node_array, sep_array, x1=x1, x2=x2, x3=x3, e1=e1, e2=e2,
             e3=e3, t=t, errs=errs, node_array=node_array,
             sep_array=sep_array)

    try:
        print('Exact solve took {} seconds for {} RHS evaluations'
              .format(fine_end - fine_start, fine_counter))
    except NameError:
        pass

    print('Approx solve took {} seconds for {} RHS evaluations'
          .format(end - start, coarse_counter))

    # Save info from the experiments
    with open(data_dir + 'info' + str(plot_num) + '.txt', 'w') as f:
        expt_info = ['distance, {}\n'.format(distance),
                     'e0, ({}, {}, {})\n'.format(ex0, ey0, ez0),
                     'order, {}\n'.format(order),
                     'adaptive, {}\n'.format(adaptive),
                     'steps, {}\n'.format(t_steps),
                     'stop, {}\n'.format(stop),
                     'exact solution, {}\n'.format(exact_solution),
                     'coarse counter, {}\n'.format(coarse_counter),
                     'coarse time, {}\n'.format(end - start)]
        try:
            expt_info += ['fine counter, {}\n'.format(fine_counter),
                          'fine time, {}\n'.format(fine_end - fine_start)]
        except NameError:
            pass

        f.writelines(expt_info)

    print('Done writing data!')


if __name__ == '__main__':
    import sys

    expt = int(sys.argv[1])
    try:
        main(expt, sys.argv[2])
    except IndexError:
        main(expt)
