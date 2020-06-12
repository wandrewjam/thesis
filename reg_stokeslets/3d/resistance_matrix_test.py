import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import solve
from force_test import assemble_quad_matrix
from sphere_integration_utils import sphere_integrate
import multiprocessing as mp


def assemble_vel_cases(sphere_nodes, shear_rate=1., shear_vec=True):
    v_array = np.tile(np.eye(3), (sphere_nodes.shape[0], 1))
    om_array = np.cross(np.eye(3)[np.newaxis, :, :],
                        sphere_nodes[:, np.newaxis, :],
                        axisc=1).reshape((-1, 3))

    if shear_vec:
        shear_vec = np.zeros(shape=(3*sphere_nodes.shape[0], 1))
        shear_vec[2::3, 0] = -sphere_nodes[:, 0] * shear_rate
        return np.hstack([v_array, om_array, shear_vec])
    else:
        strain = np.zeros(shape=(3, 3, 3))
        strain[(1, 2), (2, 1), 0] = 1
        strain[(0, 2), (2, 0), 1] = 1
        strain[(0, 1), (1, 0), 2] = 1

        e_array = np.dot(strain, sphere_nodes.T).transpose((2, 0, 1))
        e_array = e_array.reshape((-1, 3))
        return np.hstack([v_array, om_array, e_array])


def generate_resistance_matrices(eps, n_nodes, a=1., b=1., domain='free',
                                 distance=0., theta=0., phi=0., shear_vec=True,
                                 proc=1, precompute_array=None):
    print('Assembling quadrature matrix for eps = {}, nodes = {}'.format(
        eps, n_nodes))
    s_matrix, weights, nodes = assemble_quad_matrix(
        eps=eps, n_nodes=n_nodes, a=a, b=b, domain=domain, distance=distance,
        theta=theta, phi=phi, proc=proc, precompute_array=precompute_array
    )
    # Solve for the forces given 6 different velocity cases
    # print('Assembling rhs for eps = {}, nodes = {}'.format(eps, n_nodes))
    rhs = assemble_vel_cases(nodes, shear_vec=shear_vec)
    rhs_cases = rhs.shape[1]
    print('Solving for forces for eps = {}, nodes = {}'.format(eps, n_nodes))
    intermediate_solve = solve(
        s_matrix, rhs, overwrite_a=True, overwrite_b=True,
        check_finite=False, assume_a='pos'
    )
    pt_forces = (intermediate_solve.T / np.repeat(weights, repeats=3)).T

    # For each velocity case, integrate velocities to get body force and
    #   body torque
    pt_forces = pt_forces.reshape((-1, 3, rhs_cases))
    # Need to use the displacement from the center of mass for torques
    centered_nodes = nodes - distance * np.array([1, 0, 0])
    pt_torques = np.cross(centered_nodes[:, :, np.newaxis], pt_forces, axis=1)
    # print('Calculating body force and torque for eps = {}, nodes = {}'.format(
    #     eps, n_nodes))
    tmp_matrix1 = np.stack([
        sphere_integrate(pt_forces[..., i], n_nodes=n_nodes, a=a, b=b)
        for i in range(rhs_cases)
    ], axis=-1)
    tmp_matrix2 = np.stack([
        sphere_integrate(pt_torques[..., i], n_nodes=n_nodes, a=a, b=b)
        for i in range(rhs_cases)
    ], axis=-1)

    # print('Constructing resistance matrices for eps = {}, nodes = {}'.format(
    #     eps, n_nodes))
    t_matrix, p_matrix, shear_f = (tmp_matrix1[:, :3], tmp_matrix1[:, 3:6],
                                   tmp_matrix1[:, 6:])
    pt_matrix, r_matrix, shear_t = (tmp_matrix2[:, :3], tmp_matrix2[:, 3:6],
                                    tmp_matrix2[:, 6:])
    print('Finished eps={}, n_nodes={}'.format(eps, n_nodes))
    return t_matrix, p_matrix, pt_matrix, r_matrix, shear_f, shear_t


def main(proc=1, a=1., b=1., domain='free', distance=0, server='mac'):
    if server == 'linux':
        eps = [0.1, 0.05, 0.01]
        n_nodes = [12, 24, 36, 48]
    elif server == 'mac':
        eps = [0.1, 0.05]
        n_nodes = [4, 8]
    else:
        raise ValueError('\'server\' variable is not valid')

    if proc == 1:
        matrices = [
            generate_resistance_matrices(e, n, a, b, domain=domain, distance=distance, shear_vec=False)
            for e in eps for n in n_nodes
        ]
    else:
        pool = mp.Pool(processes=proc)
        result = [
            pool.apply_async(generate_resistance_matrices,
                             args=(e, n),
                             kwds={'a': a, 'b': b, 'domain': domain,
                                   'distance': distance, 'shear_vec': False})
            for e in eps for n in n_nodes
        ]
        matrices = [res.get() for res in result]

    T = np.zeros(shape=(len(eps) * 3, len(n_nodes) * 3 + 1))
    R = np.zeros(shape=(len(eps) * 3, len(n_nodes) * 3 + 1))
    E = np.zeros(shape=(len(eps) * 3, len(n_nodes) * 3 + 1))

    T[::3, 0] = eps
    R[::3, 0] = eps
    E[::3, 0] = eps

    for i in range(len(eps)):
        for j in range(len(n_nodes)):
            T[3*i:3*(i+1), 3*j+1:3*(j+1)+1] = matrices[len(n_nodes) * i + j][0]
            R[3*i:3*(i+1), 3*j+1:3*(j+1)+1] = matrices[len(n_nodes) * i + j][3]
            E[3*i:3*(i+1), 3*j+1:3*(j+1)+1] = matrices[len(n_nodes) * i + j][5]

    header = 'eps'
    for n in n_nodes:
        header += ' {}'.format(n)

    # Save full matrices in major_axis text file
    np.savetxt('t_matrix_data_{}_{}.dat'.format(a, b), T, header=header)
    np.savetxt('r_matrix_data_{}_{}.dat'.format(a, b), R, header=header)
    np.savetxt('e_matrix_data_{}_{}.dat'.format(a, b), E, header=header)


if __name__ == '__main__':
    import sys
    kwargs = {
        'a': float(sys.argv[1]),
        'b': float(sys.argv[2]),
        'domain': sys.argv[3],
        'distance': float(sys.argv[4]),
        'server': sys.argv[5]
    }

    main(**kwargs)
