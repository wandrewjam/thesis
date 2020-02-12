import numpy as np
from itertools import product
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
import cProfile


def phi(xi, eta, patch):
    """Convert from local patch coordinates to Cartesian coordinates

    Parameters
    ----------
    xi
    eta
    patch

    Returns
    -------

    """
    xi = np.array(xi)
    eta = np.array(eta)
    assert type(patch) == int

    xx = np.tan(xi)
    yy = np.tan(eta)
    dsq = np.sqrt(1 + xx**2 + yy**2)

    if patch == 1:
        x = 1. / dsq
        y = xx / dsq
        z = yy / dsq
    elif patch == 2:
        x = -xx / dsq
        y = 1. / dsq
        z = yy / dsq
    elif patch == 3:
        x = -1. / dsq
        y = -xx / dsq
        z = yy / dsq
    elif patch == 4:
        x = xx / dsq
        y = -1. / dsq
        z = yy / dsq
    elif patch == 5:
        x = -yy / dsq
        y = xx / dsq
        z = 1. / dsq
    elif patch == 6:
        x = yy / dsq
        y = xx / dsq
        z = -1. / dsq
    else:
        raise ValueError('patch must be an integer between 1 and 6, inclusive')

    return x, y, z


def geom_weights(xi, eta):
    """Calculate geometric weight at local coordinates

    Parameters
    ----------
    xi
    eta

    Returns
    -------

    """
    xi = np.array(xi)
    eta = np.array(eta)

    return (((1 + np.tan(xi)**2) * (1 + np.tan(eta)**2))
            / np.sqrt((1 + np.tan(xi)**2 + np.tan(eta)**2)**3))


def stokeslet_integrand(x_tuple, center, eps):
    """Evaluate the regularized Stokeslet located at center

    This function returns the ith component of the free-space
    regularized Stokeslet centered at center, evaluated at position x.

    Parameters
    ----------
    eps
    center
    x_tuple

    Returns
    -------

    """

    x_array = np.array(x_tuple).transpose((1, 2, 0))

    del_x = x_array - center
    r2 = np.sum(del_x**2, axis=-1)

    # Use some wild numpy broadcasting
    stokeslet = ((np.eye(3)[np.newaxis, np.newaxis, :, :]
                 * (r2[:, :, np.newaxis, np.newaxis] + 2*eps**2)
                 + del_x[:, :, :, np.newaxis] * del_x[:, :, np.newaxis, :])
                 / np.sqrt((r2[:, :, np.newaxis, np.newaxis] + eps**2)**3))
    force = -3. / 2 * np.array([0, 0, 1])
    output = np.dot(stokeslet, force)
    # for k in range(x_flat.shape[1]):
    #     x_curr = x_flat[:, k]
    #     del_x = x_curr - center
    #     r2 = np.dot(del_x, del_x)
    #
    #     stokeslet = (np.eye(3) * (r2 + 2*eps**2) / ((r2 + eps**2)**(3/2))
    #                  + np.outer(del_x, del_x) / ((r2 + eps**2)**(3/2)))
    #     force = -3. / 2 * np.array([0, 0, 1])
    #     output.append(np.dot(stokeslet, force)[i])
    return -output / (8 * np.pi)


def l2_error(x_tuple, n_nodes):
    original_shape = x_tuple[0].shape
    x_array = np.array(x_tuple)
    x_array = x_array.reshape((3, -1)).T

    assert np.linalg.norm(np.linalg.norm(x_array, axis=1) - 1) < 100*np.finfo(float).eps

    surface_velocity = list()
    for point in x_array:
        def f(x):
            return stokeslet_integrand(x, point, 0.01)

        surface_velocity.append(sphere_integrate(f, n_nodes))

    surface_velocity = np.array(surface_velocity)
    error = np.sqrt((surface_velocity[:, -1] - 1)**2)
    return error.reshape(original_shape)


def one_function(x_tuple):
    """Return 1 for each evaluation point"""
    assert x_tuple[0].shape == x_tuple[1].shape
    assert x_tuple[1].shape == x_tuple[2].shape
    return np.ones(shape=x_tuple[0].shape)


def main():
    n_nodes = 16
    eps = 0.01

    def f(x):
        return stokeslet_integrand(x, np.array([1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)]), eps)

    integral = sphere_integrate(f, n_nodes)

    def error(x):
        return l2_error(x, n_nodes)

    err = sphere_integrate(error, n_nodes)

    print(integral)
    print(err)

    # To do: test convergence


def sphere_integrate(integrand, n_nodes=16):
    eta_mesh, xi_mesh = generate_grid(n_nodes)[:2]

    del_xi = xi_mesh[1] - xi_mesh[0]
    del_eta = eta_mesh[1] - eta_mesh[0]

    patch_integrals = list()
    for i in range(6):
        patch = i + 1
        g_matrix = geom_weights(xi_mesh[:, np.newaxis],
                                eta_mesh[np.newaxis, :])

        cartesian_coordinates = phi(xi_mesh[:, np.newaxis],
                                    eta_mesh[np.newaxis, :], patch)

        f_matrix = integrand(cartesian_coordinates)

        c_matrix = np.ones(shape=xi_mesh.shape + eta_mesh.shape)
        c_matrix[(0, n_nodes), 1:n_nodes] = 1. / 2
        c_matrix[1:n_nodes, (0, n_nodes)] = 1. / 2
        c_matrix[0:n_nodes + 1:n_nodes, 0:n_nodes + 1:n_nodes] = 1. / 3
        if f_matrix.ndim == 3:
            c_matrix = c_matrix[:, :, np.newaxis]
            g_matrix = g_matrix[:, :, np.newaxis]
        patch_integrals.append(np.sum(f_matrix * c_matrix * g_matrix,
                                      axis=(0, 1)) * del_xi * del_eta)
    return np.sum(patch_integrals, axis=0)


def generate_grid(n_nodes):
    assert n_nodes % 2 == 0  # n_nodes should be even
    xi_mesh = np.linspace(-np.pi / 4, np.pi / 4, num=n_nodes + 1)
    eta_mesh = np.linspace(-np.pi / 4, np.pi / 4, num=n_nodes + 1)
    cube_nodes = list()
    for i in range(6):
        patch = i + 1
        nodes = np.linspace(-1, 1, num=n_nodes + 1)
        if patch == 1:
            cube_nodes += product([1.], nodes, nodes)
        elif patch == 2:
            cube_nodes += product(nodes, [1.], nodes)
        elif patch == 3:
            cube_nodes += product([-1.], nodes, nodes)
        elif patch == 4:
            cube_nodes += product(nodes, [-1.], nodes)
        elif patch == 5:
            cube_nodes += product(nodes, nodes, [1.])
        elif patch == 6:
            cube_nodes += product(nodes, nodes, [-1.])
    cube_nodes = np.array(list(set(cube_nodes)))
    assert cube_nodes.shape[0] == 6 * n_nodes ** 2 + 2

    sphere_nodes = cube_nodes / np.linalg.norm(cube_nodes,
                                               axis=-1)[:, np.newaxis]
    return eta_mesh, xi_mesh, sphere_nodes


if __name__ == '__main__':
    cProfile.run('main()')
