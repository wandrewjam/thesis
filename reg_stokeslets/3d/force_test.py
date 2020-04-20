import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sphere_integration_utils import (generate_grid, geom_weights,
                                      sphere_integrate, stokeslet_integrand,
                                      h1, h2, d1, d2, h1p, h2p)
import cProfile


def find_column(n_nodes, center, force_center, k, eps):
    def force(x_coords):
        """Generates a unit force at force_center, in component k"""
        output = np.zeros(shape=x_coords.shape)
        output[..., k] = np.linalg.norm(
            x_coords - force_center, axis=-1
        ) < 100 * np.finfo(float).eps
        return output

    return sphere_integrate(stokeslet_integrand, n_nodes=n_nodes, center=center, force=force, eps=eps)


def main(proc=1):
    assert proc > 0, 'proc must be a positive integer'
    assert type(proc) is int, 'proc must be a positive integer'

    eps = 0.01
    top = 5

    l2_error = np.zeros(shape=(top, 2))
    for i in range(top):
        n_nodes = 2**(i + 1)
        l2_error[i, 0] = find_solve_error(eps, n_nodes)
        l2_error[i, 1] = n_nodes

        np.savetxt('find_solve_error.txt', l2_error)


def find_solve_error(eps, n_nodes):
    a_matrix, sphere_nodes = assemble_quad_matrix(eps, n_nodes)

    unit_vel = np.tile([0, 0, -1], sphere_nodes.shape[0])
    est_force2 = np.linalg.solve(a_matrix, unit_vel)
    est_force2 = est_force2.reshape((-1, 3))

    total_force = sphere_integrate(est_force2, n_nodes=n_nodes)
    error = np.abs(total_force[2] - 6 * np.pi)

    return error


def assemble_quad_matrix(eps, n_nodes, a=1., b=1., domain='free'):
    xi_mesh, eta_mesh, nodes, ind_map = generate_grid(n_nodes, a=a, b=b)

    c_matrix = np.ones(shape=ind_map.shape[:2])
    c_matrix[(0, n_nodes), 1:n_nodes] = 1. / 2
    c_matrix[1:n_nodes, (0, n_nodes)] = 1. / 2
    c_matrix[0:n_nodes + 1:n_nodes, 0:n_nodes + 1:n_nodes] = 1. / 3

    weight_array = [
        c_matrix * geom_weights(xi_mesh[:, np.newaxis],
                                eta_mesh[np.newaxis, :], a=a, b=b, patch=patch)
        for patch in range(1, 7)
    ]

    weight_array = np.stack(weight_array, axis=-1)
    weights = np.array([sum(weight_array[ind_map == i])
                        for i in range(nodes.shape[0])])

    del_xi = xi_mesh[1] - xi_mesh[0]
    del_eta = eta_mesh[1] - eta_mesh[0]
    stokeslet = generate_stokeslet(eps, nodes, domain)
    a_matrix = -stokeslet * weights[:, np.newaxis, np.newaxis]
    a_matrix = a_matrix.transpose((0, 2, 1, 3)).reshape(
        (nodes.size, nodes.size)) * del_xi * del_eta
    return a_matrix, nodes


def ss(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    stokeslet = ((np.eye(3)[np.newaxis, np.newaxis, :, :] * h1(r, eps)
                  + del_x[:, :, :, np.newaxis] * del_x[:, :, np.newaxis, :]
                  * h2(r, eps)))
    assert np.all(stokeslet == stokeslet.transpose((1, 0, 3, 2)))
    return stokeslet


def pd(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    dipole = ((np.eye(3)[np.newaxis, np.newaxis, :, :] * d1(r, eps)
               + del_x[:, :, :, np.newaxis] * del_x[:, :, np.newaxis, :]
               * d2(r, eps)))
    assert np.all(dipole == dipole.transpose((1, 0, 3, 2)))
    return dipole


def sd(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    e1 = np.array([1, 0, 0])
    e1i = e1[np.newaxis, np.newaxis, :, np.newaxis]
    e1k = e1[np.newaxis, np.newaxis, np.newaxis, :]

    x1 = del_x[:, :, 0, np.newaxis]
    xi = del_x[:, :, :, np.newaxis]
    xk = del_x[:, :, np.newaxis, :]

    ii = np.eye(3)[np.newaxis, np.newaxis, :, :]
    doublet = ((xi * e1k + ii * x1) * h2(r, eps) + e1i * xk * h1p(r, eps) / r
               + x1 * xi * xk * h2p(r, eps) / r)
    return doublet


def rt(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    ii = np.eye(3)[np.newaxis, np.newaxis, :, :]
    xk = del_x[:, :, np.newaxis, :]
    ejk1 = np.zeros(shape=(3, 3))
    ejk1[1, 2] = 1.
    ejk1[2, 1] = -1.
    rotlet = (h1p(r, eps) / r + h2(r, eps)) * np.cross(ii, xk)
    return rotlet


def generate_stokeslet(eps, nodes, type):
    xe = nodes
    x0 = nodes
    stokeslet = ss(eps, xe, x0)
    if type == 'free':
        return stokeslet
    elif type == 'wall':
        h = xe[0]
        x_im = np.array([-1, 1, 1]) * xe
        im_stokeslet = -ss(eps, xe, x_im)

        mod_matrix = np.diag([1, -1, -1])
        dipole = np.dot(pd(eps, xe, x0), mod_matrix)

        # Still need to write Stokeslet doublet and rotlet
        doublet = np.dot(sd(eps, xe, x0), mod_matrix)
        rotlet = rt(eps, xe, x0)

        ejk1 = np.zeros(shape=(3, 3))
        ejk1[1, 2] = 1
        ejk1[2, 1] = -1

        return (stokeslet + im_stokeslet - h**2 * dipole + 2 * h * doublet
                + 2 * h * np.dot(rotlet, ejk1))


if __name__ == '__main__':
    # cProfile.run('main()')
    main(1)
