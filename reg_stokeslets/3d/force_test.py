import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sphere_integration_utils import (generate_grid, geom_weights,
                                      sphere_integrate, stokeslet_integrand,
                                      h1, h2, d1, d2, h1p, h2p,
                                      wall_stokeslet_integrand)
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


def assemble_quad_matrix(eps, n_nodes, a=1., b=1., domain='free', distance=0., theta=0., phi=0., proc=1):
    xi_mesh, eta_mesh, nodes, ind_map = generate_grid(n_nodes, a=a, b=b)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    rot_matrix = np.array([[cp, -sp, 0],
                           [ct * sp, ct * cp, -st],
                           [st * sp, st * cp, ct]])
    nodes = np.dot(rot_matrix, nodes.T).T
    nodes[:, 0] += distance
    assert domain == 'free' or np.all(nodes[:, 0] > 0)

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
    stokeslet = generate_stokeslet(eps, nodes, domain, chunks=proc)
    a_matrix = -stokeslet * weights[:, np.newaxis, np.newaxis]
    a_matrix = a_matrix.transpose((0, 2, 1, 3)).reshape(
        (nodes.size, nodes.size)) * del_xi * del_eta
    return a_matrix, nodes

# @profile
def ss(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    # assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    tmp_arr1 = h1(r, eps)
    tmp_arr2 = h2(r, eps)
    stokeslet = ((np.eye(3)[np.newaxis, np.newaxis, :, :] * tmp_arr1
                  + del_x[:, :, :, np.newaxis] * del_x[:, :, np.newaxis, :]
                  * tmp_arr2))
    assert np.all(stokeslet == stokeslet.transpose((0, 1, 3, 2)))
    return stokeslet

# @profile
def pd(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    # assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    tmp_arr1 = d1(r, eps)
    tmp_arr2 = d2(r, eps)
    dipole = ((np.eye(3)[np.newaxis, np.newaxis, :, :] * tmp_arr1
               + del_x[:, :, :, np.newaxis] * del_x[:, :, np.newaxis, :]
               * tmp_arr2))
    assert np.all(dipole == dipole.transpose((0, 1, 3, 2)))
    return dipole

# @profile
def sd(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    # assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    e1 = np.array([1, 0, 0])
    e1i = e1[np.newaxis, np.newaxis, :, np.newaxis]
    e1k = e1[np.newaxis, np.newaxis, np.newaxis, :]

    x1 = del_x[..., 0]
    x1 = x1[..., np.newaxis, np.newaxis]
    xi = del_x[..., np.newaxis]
    xk = del_x[:, :, np.newaxis, :]

    ii = np.eye(3)[np.newaxis, np.newaxis, :, :]
    tmp_arr1 = h2(r, eps)
    tmp_arr2 = h1p(r, eps)
    tmp_arr3 = h2p(r, eps)
    doublet = ((xi * e1k + ii * x1) * tmp_arr1 + e1i * xk * tmp_arr2 / r
               + x1 * xi * xk * tmp_arr3 / r)
    return doublet

# @profile
def rt(eps, xe, x0):
    del_x = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r2 = np.sum(del_x**2, axis=-1)
    # assert np.all(r2 == r2.T)
    r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    ii = np.eye(3)[np.newaxis, np.newaxis, :, :]
    xk = del_x[:, :, np.newaxis, :]
    ejk1 = np.zeros(shape=(3, 3))
    ejk1[1, 2] = 1.
    ejk1[2, 1] = -1.
    tmp_arr1 = h1p(r, eps)
    tmp_arr2 = h2(r, eps)

    cross_arr = np.cross(ii, xk)

    rotlet = (tmp_arr1 / r + tmp_arr2) * cross_arr
    return rotlet

# @profile
def generate_stokeslet(eps, nodes, type, chunks=1):
    """Generate a S x S x 3 x 3 array of Stokeslet strengths at 'nodes'
    S_{ijkl} is the (k,l) component of the Stokeslet centered at x_j
    and evaluated at x_i

    Parameters
    ----------
    chunks
    eps
    nodes
    type

    Returns
    -------

    """
    x0 = nodes
    if chunks == 1:
        xe = nodes
        stokeslet = stokeslet_helper(eps, xe, x0, type)
    else:
        xe_list = np.array_split(nodes, chunks)

        pool = mp.Pool(processes=chunks)
        result = [pool.apply_async(stokeslet_helper, args=(eps, xe, x0, type))
                  for xe in xe_list]

        result = [res.get() for res in result]
        stokeslet = np.concatenate(result)

    return stokeslet


def stokeslet_helper(eps, xe, x0, type):
    stokeslet = ss(eps, xe, x0)
    if type == 'free':
        return stokeslet
    elif type == 'wall':
        h = x0[:, 0]
        h = h[np.newaxis, :, np.newaxis, np.newaxis]
        x_im = np.array([-1, 1, 1]) * x0
        im_stokeslet = -ss(eps, xe, x_im)

        mod_matrix = np.diag([1, -1, -1])
        tmp_dip = pd(eps, xe, x_im)
        dipole = np.dot(tmp_dip, mod_matrix)

        tmp_doub = sd(eps, xe, x_im)
        doublet = np.dot(tmp_doub, mod_matrix)

        tmp_rot = rt(eps, xe, x_im)

        ejk1 = np.zeros(shape=(3, 3))
        ejk1[1, 2] = 1
        ejk1[2, 1] = -1

        rotlet = np.dot(tmp_rot, ejk1)

        return (stokeslet + im_stokeslet - h ** 2 * dipole + 2 * h * doublet
                - 2 * h * rotlet)


if __name__ == '__main__':
    # cProfile.run('main()')
    main(1)
