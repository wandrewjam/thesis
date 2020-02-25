import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sphere_integration_utils import generate_grid, sphere_integrate, stokeslet_integrand
import cProfile


def find_column(n_nodes, center, force_center, k, eps):
    def force(x_coords):
        """Generates a unit force at force_center, in component k"""
        output = np.zeros(shape=x_coords.shape)
        output[..., k] = np.linalg.norm(
            x_coords - force_center, axis=-1
        ) < 100 * np.finfo(float).eps
        return output

    return sphere_integrate(stokeslet_integrand, n_nodes=n_nodes,
                            center=center, force=force, eps=eps)


def main(proc=1):
    assert proc > 0, 'proc must be a positive integer'
    assert type(proc) is int, 'proc must be a positive integer'
    eps = 0.1
    top = 2

    l2_error = np.zeros(shape=(top, 2))
    for i in range(top):
        n_nodes = 2**(i + 1)
        l2_error[i, 0] = find_solve_error(eps, n_nodes, proc)
        l2_error[i, 1] = n_nodes

        np.savetxt('find_solve_error.txt', l2_error)


def find_solve_error(eps, n_nodes, proc):
    sphere_nodes = generate_grid(n_nodes)[2]
    total_nodes = sphere_nodes.shape[0]
    a_matrix = np.zeros(shape=(3 * total_nodes, 3 * total_nodes))
    pool = mp.Pool(processes=proc)
    for i in range(total_nodes):
        for k in range(3):
            if proc == 1:
                column = [
                    find_column(n_nodes=n_nodes, center=point,
                                force_center=sphere_nodes[i], k=k, eps=eps)
                    for point in sphere_nodes
                ]
            else:
                col_result = [pool.apply_async(
                    find_column, kwds={
                        'n_nodes': n_nodes, 'center': point,
                        'force_center': sphere_nodes[i], 'k': k, 'eps': eps
                    }) for point in sphere_nodes]
                column = [res.get() for res in col_result]
            a_matrix[:, 3 * i + k] = np.concatenate(column)
    unit_vel = np.tile([0, 0, -1], total_nodes)
    est_force = np.linalg.lstsq(a_matrix, unit_vel)[0]
    est_force = est_force.reshape((-1, 3))

    total_force = sphere_integrate(est_force, n_nodes=n_nodes)
    error = np.abs(total_force[2] - 6 * np.pi)

    return error


if __name__ == '__main__':
    # cProfile.run('main()')
    main(3)
