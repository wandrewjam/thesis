import numpy as np
import multiprocessing as mp


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


def generate_grid(n_nodes):
    """Generate a uniform mesh in patch coordinates xi and eta

    This function only generates the mesh for a single patch. The patch
    coordinate mesh is identical for each patch, the difference in
    patches is only taken into account in the mapping back to Cartesian
    coordinates.

    Parameters
    ----------
    n_nodes

    Returns
    -------

    """
    assert n_nodes % 2 == 0  # n_nodes should be even
    xi_mesh = np.linspace(-np.pi / 4, np.pi / 4, num=n_nodes + 1)
    eta_mesh = np.linspace(-np.pi / 4, np.pi / 4, num=n_nodes + 1)
    sphere_nodes = list()
    for i in range(6):
        patch = i + 1

        patch_nodes = phi(xi_mesh[:, np.newaxis], eta_mesh[np.newaxis, :],
                          patch=patch)

        patch_nodes = np.array(patch_nodes).transpose(
            (1, 2, 0))

        if patch == 1 or patch == 3:
            patch_nodes = patch_nodes.reshape((-1, 3))
            sphere_nodes.append(patch_nodes)
        elif patch == 2 or patch == 4:
            patch_nodes = patch_nodes[1:-1].reshape((-1, 3))
            sphere_nodes.append(patch_nodes)
        elif patch == 5 or patch == 6:
            patch_nodes = patch_nodes[1:-1, 1:-1].reshape((-1, 3))
            sphere_nodes.append(patch_nodes)

    sphere_nodes = np.concatenate(sphere_nodes)
    assert sphere_nodes.shape[0] == 6 * n_nodes ** 2 + 2

    return eta_mesh, xi_mesh, sphere_nodes


def stokeslet_integrand(x_tuple, center, eps, force):
    """Evaluate the regularized Stokeslet located at center

    This function returns the ith component of the free-space
    regularized Stokeslet centered at center, evaluated at position
    x_tuple.

    Parameters
    ----------
    force
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
    output = np.zeros(shape=stokeslet.shape[:3])
    f_array = force(x_array)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j, :] = np.dot(stokeslet[i, j, :, :],
                                     f_array[i, j, :])

    return -output / (8 * np.pi)


def l2_error(x_tuple, n_nodes, proc=1, eps=0.01):
    assert type(proc) is int
    assert type(n_nodes) == np.int64 or type(n_nodes) is int
    assert n_nodes % 2 == 0
    original_shape = x_tuple[0].shape
    x_array = np.array(x_tuple)
    x_array = x_array.reshape((3, -1)).T

    assert (np.linalg.norm(np.linalg.norm(x_array, axis=1) - 1)
            < 100*np.finfo(float).eps)

    def point_force(x_array):
        output = np.zeros(shape=x_array.shape)
        output[..., 2] = -3./2
        return output

    if proc == 1:
        surface_velocity = [
            sphere_integrate(stokeslet_integrand, n_nodes=n_nodes,
                             center=point, eps=eps, force=point_force)
            for point in x_array
        ]

    elif proc > 1:
        pool = mp.Pool(processes=proc)
        mp_result = [
            pool.apply_async(
                sphere_integrate, args=(stokeslet_integrand, n_nodes),
                kwds={'center': point, 'eps': eps, 'force': point_force})
            for point in x_array
        ]
        surface_velocity = [res.get() for res in mp_result]

    else:
        raise ValueError('proc must be a positive integer')

    surface_velocity = np.array(surface_velocity)
    error = np.sqrt((surface_velocity[:, -1] - 1)**2)
    return error.reshape(original_shape)


def one_function(x_tuple):
    """Return 1 for each evaluation point"""
    assert x_tuple[0].shape == x_tuple[1].shape
    assert x_tuple[1].shape == x_tuple[2].shape
    return np.ones(shape=x_tuple[0].shape)


def sphere_integrate(integrand, n_nodes=16, **kwargs):
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

        f_matrix = integrand(cartesian_coordinates, **kwargs)

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
