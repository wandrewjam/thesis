import numpy as np
from scipy.linalg import solve
import multiprocessing as mp
from sphere_integration_utils import (generate_grid, geom_weights,
                                      sphere_integrate, stokeslet_integrand,
                                      compute_helper_funs,
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
    s_matrix, weights, sphere_nodes = assemble_quad_matrix(eps, n_nodes)

    unit_vel = np.tile([0, 0, -1], sphere_nodes.shape[0])
    intermediate_solve = solve(
        s_matrix, unit_vel, overwrite_a=True, overwrite_b=True,
        check_finite=False, assume_a='pos'
    )

    est_force2 = intermediate_solve / np.repeat(weights, repeats=3)
    est_force2 = est_force2.reshape((-1, 3))

    total_force = sphere_integrate(est_force2, n_nodes=n_nodes)
    error = np.abs(total_force[2] - 6 * np.pi)

    return error


def assemble_quad_matrix(eps, n_nodes, a=1., b=1., domain='free', distance=0.,
                         theta=0., phi=0., proc=1, precompute_array=None):
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
    stokeslet = generate_stokeslet(eps, nodes, domain, chunks=proc,
                                   precompute_array=precompute_array)

    ind_u = np.triu_indices(nodes.shape[0])
    s_array = np.zeros(shape=(nodes.shape[0], nodes.shape[0], 3, 3))
    s_array[ind_u] = stokeslet
    s_matrix = s_array.transpose((0, 2, 1, 3)).reshape(
        (nodes.size, nodes.size)) * del_eta * del_xi
    return s_matrix, weights, nodes


# @profile
def ss(eps, del_x, h_arr=None, r=None, outer=None):
    if outer is None:
        outer = del_x[:, :, np.newaxis] * del_x[:, np.newaxis, :]

    if r is None:
        r2 = np.sum(del_x**2, axis=-1)
        # assert np.all(r2 == r2.T)
        r = np.sqrt(r2)[:, np.newaxis, np.newaxis]

    if h_arr is None:
        h1_arr, h2_arr = compute_helper_funs(r, eps=eps, funs=('h1', 'h2'))
    else:
        h1_arr = h_arr[0]
        h2_arr = h_arr[1]

    stokeslet = ((np.eye(3)[np.newaxis, :, :] * h1_arr
                  + outer * h2_arr))
    # assert np.all(stokeslet == stokeslet.transpose((0, 1, 3, 2)))
    return stokeslet


# @profile
def pd(eps, del_x, d_arr=None, r=None, outer=None):
    if outer is None:
        outer = del_x[:, :, np.newaxis] * del_x[:, np.newaxis, :]

    if r is None:
        r2 = np.sum(del_x**2, axis=-1)
        # assert np.all(r2 == r2.T)
        r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    if d_arr is None:
        d1_arr, d2_arr = compute_helper_funs(r, eps=eps, funs=('d1', 'd2'))
    else:
        d1_arr = d_arr[0]
        d2_arr = d_arr[1]

    dipole = ((np.eye(3)[np.newaxis, :, :] * d1_arr
               + outer * d2_arr))
    # assert np.all(dipole == dipole.transpose((0, 1, 3, 2)))
    return dipole

# @profile
def sd(eps, del_x, h_arr=None, r=None):
    e1 = np.array([1, 0, 0])
    e1i = e1[np.newaxis, :, np.newaxis]
    e1k = e1[np.newaxis, np.newaxis, :]

    x1 = del_x[..., 0]
    x1 = x1[..., np.newaxis, np.newaxis]
    xi = del_x[..., np.newaxis]
    xk = del_x[:, np.newaxis, :]

    ii = np.eye(3)[np.newaxis, :, :]

    if r is None:
        r2 = np.sum(del_x**2, axis=-1)
        # assert np.all(r2 == r2.T)
        r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    if h_arr is None:
        h2_arr, h1p_arr, h2p_arr = \
            compute_helper_funs(r, eps=eps, funs=('h2', 'h1p', 'h2p'))
    else:
        h2_arr = h_arr[0]
        h1p_arr = h_arr[1]
        h2p_arr = h_arr[2]
    doublet = ((xi * e1k + ii * x1) * h2_arr + (e1i * h1p_arr
               + x1 * xi * h2p_arr) * xk / r)
    return doublet

# @profile
def rt(eps, del_x, h_arr=None, r=None):
    ii = np.eye(3)[np.newaxis, :, :]
    xk = del_x[:, np.newaxis, :]

    if r is None:
        r2 = np.sum(del_x**2, axis=-1)
        # assert np.all(r2 == r2.T)
        r = np.sqrt(r2)[:, :, np.newaxis, np.newaxis]

    if h_arr is None:
        h2_arr, h1p_arr = compute_helper_funs(r, eps=eps, funs=('h2', 'h1p'))
    else:
        h1p_arr = h_arr[0]
        h2_arr = h_arr[1]

    cross_arr = np.cross(ii, xk)

    rotlet = (h1p_arr / r + h2_arr) * cross_arr
    return rotlet


# @profile
def generate_stokeslet(eps, nodes, type, chunks=1, precompute_array=None):
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
    ind_u = np.triu_indices(nodes.shape[0])
    del_x0 = nodes[ind_u[0]] - nodes[ind_u[1]]
    del_x = nodes[ind_u[0]] - np.array([-1, 1, 1]) * nodes[ind_u[1]]
    h = nodes[ind_u[1], 0]
    h = h[:, np.newaxis, np.newaxis]
    if chunks == 1:
        stokeslet = stokeslet_helper(eps, del_x0, del_x, h, type,
                                     precompute_array)
    else:
        del_x0_list = np.array_split(del_x0, chunks)
        del_x_list = np.array_split(del_x, chunks)
        h_list = np.array_split(h, chunks)

        pool = mp.Pool(processes=chunks)
        result = [
            pool.apply_async(stokeslet_helper,
                             args=(eps, del_x0, del_x, h, type,
                                   precompute_array))
            for (del_x0, del_x, h) in zip(del_x0_list, del_x_list, h_list)
        ]

        result = [res.get() for res in result]
        stokeslet = np.concatenate(result)

    return stokeslet

# @profile
def stokeslet_helper(eps, del_x0, del_x, h, type, precompute_array):
    outer0 = del_x0[:, np.newaxis, :] * del_x0[:, :, np.newaxis]
    r0 = np.linalg.norm(del_x0, axis=-1)[:, np.newaxis, np.newaxis]

    if precompute_array is not None:
        r_save = precompute_array[0]
        del_r = r_save[1] - r_save[0]  # Assumes uniform spacing
        look_up0 = np.round(r0 / del_r).astype(dtype='int')
        h1_arr0, h2_arr0 = [precompute_array[j][look_up0] for j in range(1, 3)]
    else:
        h1_arr0, h2_arr0 = compute_helper_funs(r0, eps=eps, funs=('h1', 'h2'))
    stokeslet = ss(eps, del_x0, h_arr=(h1_arr0, h2_arr0), r=r0, outer=outer0)
    if type == 'free':
        return stokeslet
    elif type == 'wall':
        outer = del_x[:, np.newaxis, :] * del_x[:, :, np.newaxis]
        r = np.linalg.norm(del_x, axis=-1)[:, np.newaxis, np.newaxis]

        if precompute_array is not None:
            look_up = np.round(r / del_r).astype(dtype='int')
            h1_arr, h2_arr, d1_arr, d2_arr, h1p_arr, h2p_arr = [
                precompute_array[j][look_up] for j in range(1, 7)
            ]
        else:
            h1_arr, h2_arr, d1_arr, d2_arr, h1p_arr, h2p_arr = \
                compute_helper_funs(r, eps=eps)

        im_stokeslet = -ss(eps, del_x, h_arr=(h1_arr, h2_arr), r=r,
                           outer=outer)

        mod_matrix = np.diag([1, -1, -1])
        tmp_dip = pd(eps, del_x, d_arr=(d1_arr, d2_arr), r=r, outer=outer)
        dipole = np.dot(tmp_dip, mod_matrix)

        tmp_doub = sd(eps, del_x, h_arr=(h2_arr, h1p_arr, h2p_arr), r=r)
        doublet = np.dot(tmp_doub, mod_matrix)

        tmp_rot = rt(eps, del_x, h_arr=(h1p_arr, h2_arr), r=r)

        ejk1 = np.zeros(shape=(3, 3))
        ejk1[1, 2] = 1
        ejk1[2, 1] = -1

        rotlet = np.dot(tmp_rot, ejk1)

        return (stokeslet + im_stokeslet - h ** 2 * dipole + 2 * h * doublet
                - 2 * h * rotlet)


if __name__ == '__main__':
    # cProfile.run('main()')
    main(1)
