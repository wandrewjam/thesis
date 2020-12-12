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
    n_min, n_max = 8, 16
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


def find_min_separation(com_dist, e_m, loc=False):
    mesh = generate_grid(n_nodes=48, a=1.5, b=0.5)[2]
    theta = np.arctan2(e_m[2], e_m[1])
    phi = np.arccos(e_m[0])
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    rot_matrix = np.array([[cp, -sp, 0],
                           [ct * sp, ct * cp, -st],
                           [st * sp, st * cp, ct]])
    mesh = np.dot(rot_matrix, mesh.T).T
    mesh[:, 0] += com_dist
    i_min = np.argmin(mesh[:, 0])
    sep = mesh[i_min, 0]
    if loc:
        return sep, mesh[i_min]
    else:
        return sep


def get_bond_lengths(bonds, receptors):
    bond_array = np.zeros(shape=(len(bonds), 3))
    for i, (n, yl, zl) in enumerate(bonds):
        bond_array[i] = receptors[int(n)] - np.array([0, yl, zl])
    lengths = np.linalg.norm(bond_array, axis=1)
    return lengths


def update_bonds(receptors, bonds, x1, x2, x3, rmat, dt, k0_on, k0_off, eta, eta_ts, lam, one_side=False):
    # Given an array of receptors, compute the binding rates
    # eta is the nondimensional sigma
    # what about receptors already involved in binding?

    true_receptors = np.dot(receptors, rmat.T)
    true_receptors += np.array([[x1, x2, x3]])

    if lam == 0:
        k_on = k0_on * np.pi / eta_ts * np.exp(-eta_ts*true_receptors[:, 0]**2)
        k_on *= 1 - np.bincount(bonds[:, 0].astype('int'),
                                minlength=true_receptors.shape[0])
    elif lam > 0:
        l = np.linspace(-.2, .2, num=21)
        ligands = (np.stack([np.zeros(shape=2*l.shape)]
                            + np.meshgrid(l, l), axis=-1).reshape(-1, 3)
                   + np.array([[0, x2, x3]]))
        pw_lengths = np.linalg.norm(true_receptors[:, None, :]
                                    - ligands[None, :, :], axis=-1)
        rl_dev = pw_lengths - lam
        if one_side:
            rl_dev *= rl_dev > 0
        pw_rates = k0_on * np.exp(-eta_ts * rl_dev ** 2)
        k_on = np.sum(pw_rates, axis=1)
        k_on *= 1 - np.bincount(bonds[:, 0].astype('int'),
                                minlength=true_receptors.shape[0])
    else:
        raise ValueError('lam must be positive')

    r1 = np.random.rand(k_on.shape[0])
    form_bonds = (1 - np.exp(-dt * k_on) > r1)

    # Form bonds
    for i, el in enumerate(form_bonds):
        if el:
            if lam == 0:
                ligand = (np.sqrt(1 / eta_ts)*np.random.randn(2)
                          + true_receptors[i, 1:])
            elif lam > 0:
                r3 = np.random.rand()
                ligand_id = np.searchsorted(np.cumsum(pw_rates[i, :]),
                                            r3 * k_on[i])
                ligand = ligands[ligand_id, 1:]
            bonds = np.append(bonds, np.append(i, ligand)[None, :], axis=0)

    # Break bonds
    r2 = np.random.rand(len(bonds))
    bond_lens = get_bond_lengths(bonds, true_receptors)
    rl_dev = bond_lens - lam
    if one_side:
        rl_dev *= rl_dev > 0
    k_off = k0_off * np.exp((eta - eta_ts) * rl_dev ** 2)
    break_bonds = np.nonzero(1 - np.exp(-dt * k_off) > r2)
    bonds = np.delete(bonds, break_bonds, axis=0)

    return bonds


def find_bond_forces(receptors, bonds, center, rmat, kappa, lam,
                     one_side=False):
    force, torque = np.zeros(3), np.zeros(3)
    true_receptors = np.dot(receptors, rmat.T) + center
    for bond in bonds:
        receptor = true_receptors[bond[0].astype(int)]
        ligand = np.append([0], bond[1:])
        r = np.linalg.norm(receptor - ligand)

        bond_force = (lam - r) * (receptor - ligand) / r
        bond_torque = np.cross(receptor - center,
                               (lam - r) * (receptor - ligand) / r)

        if one_side:
            bond_force *= (r > lam)
            bond_torque *= (r > lam)

        force += bond_force
        torque += bond_torque
    force *= kappa
    torque *= kappa
    return force, torque


def nondimensionalize(l_scale, shear, mu, l_sep, dimk0_on, dimk0_off, sig,
                      sig_ts, temp):
    # Units:
    #    Force - pN
    #    Length - micron
    #    Time - second

    k_b = 1.38e-5
    t_scale = 1. / shear
    f_scale = mu * l_scale**2 / t_scale
    lam = l_sep / l_scale
    k0_on, k0_off = t_scale * dimk0_on, t_scale * dimk0_off
    eta = sig * l_scale**2 / (2 * k_b * temp)
    eta_ts = sig_ts * l_scale**2 / (2 * k_b * temp)
    kappa = sig * l_scale / f_scale

    return t_scale, f_scale, lam, k0_on, k0_off, eta, eta_ts, kappa


def evaluate_motion_equations(h, e_m, forces, torques, exact_vels, a=1.0,
                              b=1.0, n_nodes=8, domain='free', proc=1,
                              save_quad_matrix=False, precompute_array=None):
    eps = eps_picker(n_nodes, a, b)
    theta = np.arctan2(e_m[2], e_m[1])
    phi = np.arccos(e_m[0])
    result = generate_resistance_matrices(
        eps, n_nodes, a=a, b=b, domain=domain, distance=h, theta=theta,
        phi=phi, proc=proc, save_quad_matrix=save_quad_matrix,
        precompute_array=precompute_array)
    if save_quad_matrix:
        (t_matrix, p_matrix, pt_matrix, r_matrix,
         shear_f, shear_t, s_matrix) = result
    else:
        t_matrix, p_matrix, pt_matrix, r_matrix, shear_f, shear_t = result

    res_matrix = np.block([[t_matrix, p_matrix], [pt_matrix, r_matrix]])
    gen_forces = np.block([[shear_f - forces.reshape((3, 1))],
                           [shear_t - torques.reshape((3, 1))]])
    gen_vels = np.linalg.solve(res_matrix, -gen_forces)
    trans_vels, rot_vels = np.squeeze(gen_vels[:3]), np.squeeze(gen_vels[3:])
    # trans_vels = (np.linalg.solve(t_matrix, forces + shear_f)
    #               + np.linalg.solve(p_matrix, torques + shear_t))
    # rot_vels = (np.linalg.solve(pt_matrix, forces + shear_f)
    #             + np.linalg.solve(r_matrix, torques + shear_t))

    exact = exact_vels(e_m)
    velocity_errors = np.concatenate((trans_vels, rot_vels)) - exact

    dx1, dx2, dx3 = trans_vels
    om1, om2, om3 = rot_vels

    # Define a counter for the number of RHS evaluations
    try:
        evaluate_motion_equations.counter += 1
    except AttributeError:
        evaluate_motion_equations.counter = 1

    if save_quad_matrix:
        return dx1, dx2, dx3, om1, om2, om3, velocity_errors, s_matrix
    else:
        return dx1, dx2, dx3, om1, om2, om3, velocity_errors


def correct_matrix(r_matrix):
    u, s, vh = np.linalg.svd(r_matrix)
    d = np.sign(np.linalg.det(np.dot(u, vh)))
    new_s = np.diag([1, 1, d])
    return np.dot(u, np.dot(new_s, vh))


def repulsive_force(com_dist, e_m):
    sep, point = find_min_separation(com_dist, e_m, loc=True)
    f0 = 1.25e9
    tau = 2000.
    rep_force = (f0 * (tau * np.exp(-tau * sep))
                 / (1 - np.exp(-tau * sep)))
    return point, rep_force


def valid_forces(old_center, old_rmat, new_center, new_rmat, receptors, bonds, kappa, lam, one_side, check_bonds=True):
    old_rep_force = repulsive_force(old_center[0], old_rmat[:, 0])[1]
    new_rep_force = repulsive_force(new_center[0], new_rmat[:, 0])[1]

    if old_rep_force < 0.1 and new_rep_force < 0.1:
        valid_rep_force = True
    else:
        valid_rep_force = np.abs(np.log10(old_rep_force / new_rep_force)) < 1

    old_forces, old_torques = find_bond_forces(receptors, bonds, old_center,
                                               old_rmat, kappa, lam, one_side)
    new_forces, new_torques = find_bond_forces(receptors, bonds, new_center,
                                               new_rmat, kappa, lam, one_side)

    if check_bonds:
        valid_bond_force = (np.linalg.norm(new_forces - old_forces) < 10.
                            and np.linalg.norm(new_torques - old_torques) < 10.)
    else:
        valid_bond_force = True

    return valid_rep_force and valid_bond_force


def time_step(dt, x1, x2, x3, r_matrix, forces, torques, exact_vels, n_nodes=8, a=1.0, b=1.0, domain='free',
              order='2nd', proc=1, save_quad_matrix_info=False, receptors=None, bonds=None, eta=1, eta_ts=1, kappa=1,
              lam=0, k0_on=1, k0_off=1, precompute_array=None, level=0, check_bonds=True, one_side=True):
    valid_test_nodes = 48

    if level >= 8:
        print('max recursions reached in time_step')
        raise OverflowError()

    # Compute the force and torque generated by existing bonds
    if receptors is not None:
        center = np.array([x1, x2, x3])
        forces, torques = find_bond_forces(receptors, bonds, center, r_matrix,
                                           kappa, lam, one_side=one_side)
        # Try a repulsive force
        point, rep_force = repulsive_force(center[0], r_matrix[:, 0])
        forces += np.array([rep_force, 0, 0])
        torques += np.cross([point[0] - center[0], point[1], point[2]],
                            [rep_force, 0, 0])

    if order == '1st':
        result = evaluate_motion_equations(
            x1, r_matrix[:, 0], forces, torques, exact_vels, a=a, b=b,
            n_nodes=n_nodes, domain=domain, proc=proc,
            save_quad_matrix=save_quad_matrix_info, precompute_array=precompute_array)

        if save_quad_matrix_info:
            dx1, dx2, dx3, om1, om2, om3, velocity_errors, s_matrix = result
            s_matrices = (s_matrix,)
        else:
            dx1, dx2, dx3, om1, om2, om3, velocity_errors = result

        new_x1 = [x1 + dt * dx1]
        new_x2 = [x2 + dt * dx2]
        new_x3 = [x3 + dt * dx3]
        om = np.array([om1, om2, om3])
        new_rmat = r_matrix + dt * np.cross(om, r_matrix, axisb=0, axisc=0)
        new_rmat = [correct_matrix(new_rmat)]

        # Check that we have a valid orientation
        if (not valid_orientation(valid_test_nodes, new_rmat[-1][:, 0],
                                  distance=new_x1[-1], a=a, b=b)
                and domain == 'wall'):
            s_matrices = tuple()
            # Then take 2 half time-steps
            result1 = time_step(dt / 2, x1, x2, x3, r_matrix, forces, torques, exact_vels, n_nodes, a, b, domain, order,
                                proc, save_quad_matrix_info, receptors, bonds, eta, eta_ts, kappa, lam=lam, k0_on=k0_on,
                                k0_off=k0_off, precompute_array=precompute_array, level=level + 1,
                                check_bonds=check_bonds, one_side=one_side)[:-1]
            tmp_x1, tmp_x2, tmp_x3, tmp_rmat = result1[:4]
            bonds = result1[6]
            dt_list = result1[7]

            if save_quad_matrix_info:
                s_matrices += result1[-1]

            half_step2 = time_step(dt / 2, tmp_x1[-1], tmp_x2[-1], tmp_x3[-1], tmp_rmat[-1], forces, torques,
                                   exact_vels, n_nodes, a, b, domain, order, proc, save_quad_matrix_info, receptors,
                                   bonds, eta, eta_ts, kappa, lam=lam, k0_on=k0_on, k0_off=k0_off,
                                   precompute_array=precompute_array, level=level + 1, check_bonds=check_bonds, one_side=one_side)
            tmp2_x1, tmp2_x2, tmp2_x3, tmp2_rmat, velocity_errors = half_step2[:5]
            new_x1 = tmp_x1 + tmp2_x1
            new_x2 = tmp_x2 + tmp2_x2
            new_x3 = tmp_x3 + tmp2_x3
            new_rmat = tmp_rmat + tmp2_rmat
            new_bonds = half_step2[6]
            dt_list += half_step2[7]
            if save_quad_matrix_info:
                s_matrices += half_step2[-1]
        else:
            if receptors is not None:
                new_bonds = update_bonds(receptors, bonds, x1, x2, x3, r_matrix, dt, k0_on, k0_off, eta, eta_ts, lam)
            else:
                new_bonds = None
            dt_list = [dt]

    elif order == '2nd':
        result = evaluate_motion_equations(
            x1, r_matrix[:, 0], forces, torques, exact_vels, a=a, b=b,
            n_nodes=n_nodes, domain=domain, proc=proc,
            save_quad_matrix=save_quad_matrix_info,
            precompute_array=precompute_array)

        if save_quad_matrix_info:
            dx1, dx2, dx3, om1, om2, om3, velocity_errors = result[:-1]
            s_matrices = (result[-1],)
        else:
            dx1, dx2, dx3, om1, om2, om3, velocity_errors = result

        prd_x1 = x1 + dt * dx1
        prd_x2 = x2 + dt * dx2
        prd_x3 = x3 + dt * dx3

        k1_mat = np.cross([om1, om2, om3], r_matrix, axisb=0, axisc=0)
        prd_rmat = r_matrix + dt * k1_mat
        prd_rmat = correct_matrix(prd_rmat)

        try:
            half_step2 = evaluate_motion_equations(
                prd_x1, prd_rmat[:, 0], forces, torques, exact_vels, a=a, b=b,
                n_nodes=n_nodes, domain=domain, proc=proc,
                save_quad_matrix=save_quad_matrix_info,
                precompute_array=precompute_array)

            if save_quad_matrix_info:
                diff_eq_rhs = half_step2[:-2]
                s_matrices += (half_step2[-1],)
            else:
                diff_eq_rhs = half_step2[:-1]

            dx1_p, dx2_p, dx3_p, om1_p, om2_p, om3_p = diff_eq_rhs

            # Need to figure out how to time step the matrix forward
            new_x1 = [x1 + dt / 2 * (dx1 + dx1_p)]
            new_x2 = [x2 + dt / 2 * (dx2 + dx2_p)]
            new_x3 = [x3 + dt / 2 * (dx3 + dx3_p)]
            k2_mat = np.cross([om1_p, om2_p, om3_p], prd_rmat,
                              axisb=0, axisc=0)
            new_rmat = r_matrix + dt/2 * (k1_mat + k2_mat)

            new_rmat = [correct_matrix(new_rmat)]

            new_center = np.array(new_x1 + new_x2 + new_x3)

            if ((not valid_orientation(valid_test_nodes, new_rmat[-1][:, 0],
                                       distance=new_x1[-1], a=a, b=b)
                    and domain == 'wall') or
                    (not valid_forces(center, r_matrix, new_center, new_rmat[-1], receptors, bonds, kappa, lam,
                                      one_side, check_bonds))):
                raise AssertionError('next step will not be valid')

            # We can only get to this code if the end position is valid
            if receptors is not None:
                new_bonds = [update_bonds(receptors, bonds, x1, x2, x3, r_matrix, dt, k0_on, k0_off, eta, eta_ts, lam)]
            else:
                new_bonds = None
            dt_list = [dt]

        except AssertionError:
            # If we get an invalid orientation, then take 2 half-steps
            s_matrices = tuple()
            result1 = time_step(dt / 2, x1, x2, x3, r_matrix, forces, torques, exact_vels, n_nodes, a, b, domain, order,
                                proc, save_quad_matrix_info, receptors, bonds, eta, eta_ts, kappa, lam, k0_on, k0_off,
                                precompute_array=precompute_array, level=level + 1, check_bonds=check_bonds, one_side=one_side)

            tmp_x1, tmp_x2, tmp_x3, tmp_rmat = result1[:4]
            new_bonds = result1[6]
            dt_list = result1[7]

            if save_quad_matrix_info:
                s_matrices += result1[-1]

            half_step2 = time_step(dt / 2, tmp_x1[-1], tmp_x2[-1], tmp_x3[-1], tmp_rmat[-1], forces, torques,
                                   exact_vels, n_nodes, a, b, domain, order, proc, save_quad_matrix_info, receptors,
                                   new_bonds[-1], eta, eta_ts, kappa, lam, k0_on, k0_off,
                                   precompute_array=precompute_array, level=level + 1, check_bonds=check_bonds, one_side=one_side)

            tmp2_x1, tmp2_x2, tmp2_x3, tmp2_rmat, velocity_errors = half_step2[:5]
            new_x1 = tmp_x1 + tmp2_x1
            new_x2 = tmp_x2 + tmp2_x2
            new_x3 = tmp_x3 + tmp2_x3
            new_rmat = tmp_rmat + tmp2_rmat

            new_bonds += half_step2[6]
            dt_list += half_step2[7]

            if save_quad_matrix_info:
                s_matrices += half_step2[-1]

    elif order == '4th':
        k1 = evaluate_motion_equations(
            x1, r_matrix[:, 0], forces, torques, exact_vels, a=a, b=b,
            n_nodes=n_nodes, domain=domain, proc=proc,
            save_quad_matrix=save_quad_matrix_info)

        k1_mat = np.cross(k1[3:6], r_matrix, axisb=0, axisc=0)
        p1_rm = r_matrix + dt / 2 * k1_mat
        p1_rm = correct_matrix(p1_rm)
        p1_x1 = x1 + dt * k1[0] / 2

        try:
            k2 = evaluate_motion_equations(
                p1_x1, p1_rm[:, 0], forces, torques, exact_vels, a=a, b=b,
                n_nodes=n_nodes, domain=domain, proc=proc,
                save_quad_matrix=save_quad_matrix_info)

            k2_mat = np.cross(k2[3:6], p1_rm, axisb=0, axisc=0)
            p2_rm = r_matrix + dt / 2 * k2_mat
            p2_rm = correct_matrix(p2_rm)
            p2_x1 = x1 + dt * k2[0] / 2

            k3 = evaluate_motion_equations(
                p2_x1, p2_rm[:, 0], forces, torques, exact_vels, a=a, b=b,
                n_nodes=n_nodes, domain=domain, proc=proc,
                save_quad_matrix=save_quad_matrix_info)

            k3_mat = np.cross(k3[3:6], p2_rm, axisb=0, axisc=0)
            p3_rm = r_matrix + dt * k3_mat
            p3_rm = correct_matrix(p3_rm)
            p3_x1 = x1 + dt * k3[0]

            k4 = evaluate_motion_equations(
                p3_x1, p3_rm[:, 0], forces, torques, exact_vels, a=a, b=b,
                n_nodes=n_nodes, domain=domain, proc=proc,
                save_quad_matrix=save_quad_matrix_info)

            velocity_errors = k4[6]

            if save_quad_matrix_info:
                s_matrices = (k1[-1], k2[-1], k3[-1], k4[-1])

            new_x1 = [x1 + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6]
            new_x2 = [x2 + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6]
            new_x3 = [x3 + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6]

            k4_mat = np.cross(k4[3:6], p3_rm, axisb=0, axisc=0)
            new_rmat = r_matrix + dt/6 * (k1_mat + 2 * k2_mat
                                          + 2 * k3_mat + k4_mat)
            new_rmat = [correct_matrix(new_rmat)]

            if (not valid_orientation(valid_test_nodes, new_rmat[:, 0],
                                      distance=new_x1, a=a, b=b)
                    and domain == 'wall'):
                raise AssertionError('next step will not be valid')

            # We can only get to this code if the end position is valid
            if receptors is not None:
                new_bonds = update_bonds(receptors, bonds, x1, x2, x3, r_matrix, dt, k0_on, k0_off, eta, eta_ts, lam)
            else:
                new_bonds = None
            dt_list = [dt]

        except AssertionError:
            # Take two half-steps
            s_matrices = tuple()

            half_step1 = time_step(dt / 2, x1, x2, x3, r_matrix[:, 0], forces, torques, exact_vels, n_nodes, a, b,
                                   domain, order, proc, save_quad_matrix_info, receptors, bonds, eta, eta_ts, kappa,
                                   lam=lam, k0_on=k0_on, k0_off=k0_off, precompute_array=precompute_array,
                                   level=level + 1, check_bonds=check_bonds, one_side=one_side)

            tmp_x1, tmp_x2, tmp_x3, tmp_rmat = half_step1[:4]
            bonds = half_step1[6]
            dt_list = half_step1[7]

            if save_quad_matrix_info:
                s_matrices += half_step1[-1]

            half_step2 = time_step(dt / 2, tmp_x1[-1], tmp_x2[-1], tmp_x3[-1], tmp_rmat[-1], forces, torques,
                                   exact_vels, n_nodes, a, b, domain, order, proc, save_quad_matrix_info, receptors,
                                   bonds, eta, eta_ts, kappa, lam=lam, k0_on=k0_on, k0_off=k0_off,
                                   precompute_array=precompute_array, level=level + 1, check_bonds=check_bonds, one_side=one_side)

            tmp2_x1, tmp2_x2, tmp2_x3, tmp2_rmat, velocity_errors = half_step2[:5]
            new_x1 = tmp_x1 + tmp2_x1
            new_x2 = tmp_x2 + tmp2_x2
            new_x3 = tmp_x3 + tmp2_x3
            new_rmat = tmp_rmat + tmp2_rmat
            new_bonds = half_step2[6]
            dt_list += half_step2[7]

            if save_quad_matrix_info:
                s_matrices += half_step2[-1]

    else:
        print('order is not valid')
        raise ValueError()

    if save_quad_matrix_info:
        return (new_x1, new_x2, new_x3, new_rmat, velocity_errors, receptors,
                new_bonds, dt_list, s_matrices)
    else:
        return (new_x1, new_x2, new_x3, new_rmat, velocity_errors,
                receptors, new_bonds, dt_list)

    # if save_quad_matrix_info:
    #     return (new_x1, new_x2, new_x3, new_rmat, velocity_errors, receptors,
    #             new_bonds, s_matrices)
    # else:
    #     return (new_x1, new_x2, new_x3, new_rmat, velocity_errors,
    #             receptors, new_bonds)


def integrate_motion(t_span, num_steps, init, exact_vels, n_nodes=None, a=1.0, b=1.0, domain='free', order='2nd',
                     adaptive=True, proc=1, forces=None, torques=None, save_quad_matrix_info=False, receptors=None,
                     bonds=None, eta=1, eta_ts=1, kappa=1, lam=0, k0_on=1, k0_off=1, check_bonds=True, one_side=True):
    # Check that we have a valid combination of n_nodes and adaptive
    assert n_nodes > 0 or adaptive

    # x1, x2, x3 = np.zeros(shape=(3, num_steps+1))
    # r_matrices = np.zeros(shape=(3, 3, num_steps+1))
    # x1[0], x2[0], x3[0] = init[:3]

    x1, x2, x3 = [[xi] for xi in init[:3]]

    theta, phi = np.arctan2(init[5], init[4]), np.arccos(init[3])
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    r_matrix = np.array([[cp, -sp, 0],
                         [ct * sp, ct * cp, -st],
                         [st * sp, st * cp, ct]])
    # r_matrices[:, :, 0] = r_matrix
    r_matrices = [r_matrix]

    t_length = t_span[1] - t_span[0]
    dt = t_length / num_steps
    if forces is None:
        forces = np.zeros((3, 1))
    if torques is None:
        torques = np.zeros((3, 1))

    receptor_history = [receptors, ]
    bond_history = [bonds, ]

    errs = np.zeros(shape=(6, num_steps+1))
    node_array, sep_array = np.zeros(shape=(2, num_steps+1))

    if save_quad_matrix_info:
        s_matrices = []

    t = [0]
    try:
        for i in range(num_steps):
            if adaptive:
                # Find the wall separation
                # sep = find_min_separation(x1[i], r_matrices[:, 0, i])
                sep = find_min_separation(x1[-1], r_matrices[-1][:, 0])
                sep_array[i] = sep

                # Pick n_nodes based on separation
                n_nodes = n_picker(sep)

            node_array[i] = n_nodes
            receptors, bonds = receptor_history[-1], bond_history[-1]
            # res = time_step(
            #     dt, x1[i], x2[i], x3[i], r_matrices[:, :, i], forces, torques,
            #     exact_vels, n_nodes=n_nodes, a=a, b=b, domain=domain,
            #     order=order, proc=proc,
            #     save_quad_matrix_info=save_quad_matrix_info,
            #     receptors=receptors, bonds=bonds, eta=eta, eta_ts=eta_ts,
            #     kappa=kappa, lam=lam, k0_on=k0_on, k0_off=k0_off)
            res = time_step(dt, x1[-1], x2[-1], x3[-1], r_matrices[-1], forces, torques, exact_vels, n_nodes=n_nodes,
                            a=a, b=b, domain=domain, order=order, proc=proc,
                            save_quad_matrix_info=save_quad_matrix_info, receptors=receptors, bonds=bonds, eta=eta,
                            eta_ts=eta_ts, kappa=kappa, lam=lam, k0_on=k0_on, k0_off=k0_off, check_bonds=check_bonds, one_side=one_side)
            if save_quad_matrix_info:
                # (x1[i+1], x2[i+1], x3[i+1], r_matrices[:, :, i+1],
                #  errs[:, i+1]) = res[:5]
                x1 += res[0]
                x2 += res[1]
                x3 += res[2]
                r_matrices += res[3]
                errs[:, i+1] = res[4]
                receptor_history.append(res[5])
                # bond_history.append(res[6])
                bond_history += res[6]
                s_matrices.append(res[8])

                new_t = np.cumsum(res[7]) + t[-1]
                t = np.append(t, new_t)
            else:
                # (x1[i+1], x2[i+1], x3[i+1], r_matrices[:, :, i+1],
                #  errs[:, i+1]) = res[:5]
                x1 += res[0]
                x2 += res[1]
                x3 += res[2]
                r_matrices += res[3]
                errs[:, i+1] = res[4]
                receptor_history.append(res[5])
                # bond_history.append(res[6])

                bond_history += res[6]
                new_t = np.cumsum(res[7]) + t[-1]
                t = np.append(t, new_t)
    except (AssertionError, OverflowError, ValueError):
        print('Encountered an error while integrating. Halting and '
              'outputting computation results so far.')
        pass

    if save_quad_matrix_info:
        return (np.array(x1), np.array(x2), np.array(x3),
                np.stack(r_matrices, axis=-1), errs, node_array, sep_array,
                receptor_history, bond_history, t, s_matrices)
    else:
        return (np.array(x1), np.array(x2), np.array(x3),
                np.stack(r_matrices, axis=-1), errs, node_array, sep_array,
                receptor_history, bond_history, t)

    # if save_quad_matrix_info:
    #     return (x1, x2, x3, r_matrices, errs, node_array, sep_array,
    #             receptor_history, bond_history, s_matrices)
    # else:
    #     return (x1, x2, x3, r_matrices, errs, node_array, sep_array,
    #             receptor_history, bond_history)


def main(plot_num, server='mac', proc=1):
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
    order = '4th'

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
            or 81 <= plot_num <= 84 or 91 <= plot_num <= 94):
        n_nodes = 8
    elif (6 <= plot_num <= 9 or 16 <= plot_num <= 19 or 26 <= plot_num <= 29
          or 36 == plot_num or 46 <= plot_num <= 49 or 56 <= plot_num <= 59
          or 70 <= plot_num <= 79 or 86 <= plot_num <= 89
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
        np.savez(data_dir + 'fine' + str(plot_num) + '_' + order, x1_fine, x2_fine,
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

        ## # Initialize counter and timer for fine simulation
        ## evaluate_motion_equations.counter = 0
        ## fine_start = timer()

        # Run the fine simulations
        if plot_num < 80:
            fine_result = integrate_motion([0., stop], t_steps, init, exact_vels, fine_nodes, a=a, b=b, domain='wall',
                                           order=order, adaptive=False, proc=proc)
        else:
            fine_result = integrate_motion([0., stop], t_steps, init, exact_vels, fine_nodes, a=a, b=b, domain='free',
                                           order=order, adaptive=False, proc=proc)
        x1_fine, x2_fine, x3_fine, em_fine = fine_result[:4]

        ## # Save the end state after the fine simulations
        ## fine_end = timer()
        ## fine_counter = evaluate_motion_equations.counter

        ## e1_fine, e2_fine, e3_fine = em_fine

        ## np.savez(data_dir + 'fine' + str(plot_num), x1_fine, x2_fine,
        ##          x3_fine, e1_fine, e2_fine, e3_fine, t, x1=x1_fine,
        ##          x2=x2_fine, x3=x3_fine, e1=e1_fine, e2=e2_fine,
        ##          e3=e3_fine, t=t)

        exact_solution = False

    if distance == 0:
        domain = 'free'
    elif distance > 0:
        domain = 'wall'
    else:
        raise ValueError('value of distance is unexpected')

    if adaptive:
        # Initialize counter and timer for adaptive simulation
        evaluate_motion_equations.counter = 0
        adapt_start = timer()

        # Run the adaptive simulations
        adapt_result = integrate_motion([0., stop], t_steps, init, exact_vels, n_nodes, a=a, b=b, domain=domain,
                                        order=order, adaptive=adaptive, proc=proc)
        (x1_adapt, x2_adapt, x3_adapt, em_adapt, errs_adapt, node_array,
         sep_array) = adapt_result

        # Save the end state after the adaptive simulations
        adapt_end = timer()
        adapt_counter = evaluate_motion_equations.counter

        e1_adapt, e2_adapt, e3_adapt = em_adapt
        np.savez(data_dir + 'adapt' + str(plot_num) + '_' + order, x1_adapt, x2_adapt,
                 x3_adapt, e1_adapt, e2_adapt, e3_adapt, t, errs_adapt,
                 node_array, sep_array, x1_adapt=x1_adapt, x2_adapt=x2_adapt,
                 x3_adapt=x3_adapt, e1_adapt=e1_adapt, e2_adapt=e2_adapt,
                 e3_adapt=e3_adapt, t=t, errs_adapt=errs_adapt,
                 node_array=node_array, sep_array=sep_array)

    # Initialize counter and timer for coarse simulation
    start = timer()
    evaluate_motion_equations.counter = 0

    # Run the coarse simulations
    coarse_result = integrate_motion([0., stop], t_steps, init, exact_vels, n_nodes, a=a, b=b, domain=domain,
                                     order=order, adaptive=False, proc=proc)
    x1, x2, x3, e_m, errs, node_array, sep_array = coarse_result[:7]

    # Save the end state after the coarse simulations
    end = timer()
    coarse_counter = evaluate_motion_equations.counter

    e1, e2, e3 = e_m
    np.savez(data_dir + 'coarse' + str(plot_num) + '_' + order, x1, x2, x3, e1, e2, e3, t,
             errs, node_array, sep_array, x1=x1, x2=x2, x3=x3, e1=e1, e2=e2,
             e3=e3, t=t, errs=errs, node_array=node_array,
             sep_array=sep_array)

    try:
        print('Exact solve took {} seconds for {} RHS evaluations'
              .format(fine_end - fine_start, fine_counter))
    except NameError:
        pass

    try:
        print('Adaptive solve took {} seconds for {} RHS evaluations'
              .format(adapt_end - adapt_start, adapt_counter))
    except NameError:
        pass

    print('Approx solve took {} seconds for {} RHS evaluations'
          .format(end - start, coarse_counter))

    # Save info from the experiments
    with open(data_dir + 'info' + str(plot_num) + '_' + order + '.txt', 'w') as f:
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
        try:
            expt_info += ['adapt counter, {}\n'.format(adapt_counter),
                          'adapt time, {}\n'.format(adapt_end - adapt_start)]
        except NameError:
            pass

        f.writelines(expt_info)

    print('Done writing data!')


if __name__ == '__main__':
    import sys

    expt = int(sys.argv[1])
    try:
        main(expt, sys.argv[2], int(sys.argv[3]))
    except IndexError:
        try:
            main(expt, sys.argv[2])
        except IndexError:
            main(expt)
    print('Done')
