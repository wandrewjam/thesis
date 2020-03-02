import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sphere_integration_utils import (generate_grid, geom_weights,
                                      sphere_integrate, stokeslet_integrand)
import cProfile
from timeit import default_timer as timer


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

    eps = 0.01
    top = 4

    l2_error = np.zeros(shape=(top, 2))
    for i in range(top):
        n_nodes = 2**(i + 1)
        l2_error[i, 0] = find_solve_error(eps, n_nodes)
        l2_error[i, 1] = n_nodes

        np.savetxt('find_solve_error.txt', l2_error)


def find_solve_error(eps, n_nodes):
    a_matrix, sphere_nodes = assemble_quad_matrix(eps, n_nodes)

    unit_vel = np.tile([0, 0, -1], sphere_nodes.shape[0])
    est_force2 = np.linalg.lstsq(a_matrix, unit_vel)[0]
    est_force2 = est_force2.reshape((-1, 3))

    total_force = sphere_integrate(est_force2, n_nodes=n_nodes)
    error = np.abs(total_force[2] - 6 * np.pi)

    return error


def assemble_quad_matrix(eps, n_nodes):
    xi_mesh, eta_mesh, sphere_nodes, ind_map = generate_grid(n_nodes)

    c_matrix = np.ones(shape=ind_map.shape[:2])
    c_matrix[(0, n_nodes), 1:n_nodes] = 1. / 2
    c_matrix[1:n_nodes, (0, n_nodes)] = 1. / 2
    c_matrix[0:n_nodes + 1:n_nodes, 0:n_nodes + 1:n_nodes] = 1. / 3
    weight_array = c_matrix * geom_weights(xi_mesh[:, np.newaxis],
                                           eta_mesh[np.newaxis, :])
    weight_array = np.tile(weight_array[:, :, np.newaxis], (1, 1, 6))
    weights = np.array([sum(weight_array[ind_map == i])
                        for i in range(sphere_nodes.shape[0])])
    del_xi = xi_mesh[1] - xi_mesh[0]
    del_eta = eta_mesh[1] - eta_mesh[0]
    del_x = sphere_nodes[:, np.newaxis, :] - sphere_nodes[np.newaxis, :, :]
    r2 = np.sum(del_x ** 2, axis=-1)
    assert sum(np.diag(r2)) == 0
    assert np.all(r2 == r2.T)
    stokeslet = ((np.eye(3)[np.newaxis, np.newaxis, :, :]
                  * (r2[:, :, np.newaxis, np.newaxis] + 2 * eps ** 2)
                  + del_x[:, :, :, np.newaxis] * del_x[:, :, np.newaxis, :])
                 / np.sqrt((r2[:, :, np.newaxis, np.newaxis] + eps ** 2) ** 3))
    assert np.all(stokeslet == stokeslet.transpose((1, 0, 3, 2)))
    a_matrix = -stokeslet * weights[:, np.newaxis, np.newaxis] / (8 * np.pi)
    a_matrix = a_matrix.transpose((0, 2, 1, 3)).reshape(
        (sphere_nodes.size, sphere_nodes.size)) * del_xi * del_eta
    return a_matrix, sphere_nodes


if __name__ == '__main__':
    # cProfile.run('main()')
    main(3)
