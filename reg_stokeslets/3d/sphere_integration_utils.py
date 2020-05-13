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


def phi_deriv(xi, eta, patch):
    """Find the tangent vector at (xi, eta) in the xi direction

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

    xx = np.tan(xi) * np.ones(shape=eta.shape)
    yy = np.tan(eta) * np.ones(shape=xi.shape)

    cc = 1 + xx**2
    dd = 1 + yy**2
    dsq = np.sqrt(1 + xx**2 + yy**2)

    coeff1 = (cc / dsq ** 3)[:, :, np.newaxis]
    coeff2 = (dd / dsq ** 3)[:, :, np.newaxis]
    xi_reference = coeff1 * np.stack([-xx, dd, -xx * yy], axis=-1)
    eta_reference = coeff2 * np.stack([-yy, -xx*yy, cc], axis=-1)

    if patch == 1:
        # xi_deriv = cc / dsq**3 * np.stack([-xx, dd, -xx*yy], axis=-1)
        # xi_deriv = np.array([1, 1, 1]) * xi_reference[..., (0, 1, 2)]
        # eta_deriv = dd / dsq**3 * np.stack([-yy, -xx*yy, cc], axis=-1)
        # eta_deriv = np.array([1, 1, 1]) * eta_reference[..., (0, 1, 2)]
        sign = np.ones(3)
        order = (0, 1, 2)
    elif patch == 2:
        sign = np.array([-1, 1, 1])
        order = (1, 0, 2)
    elif patch == 3:
        sign = np.array([-1, -1, 1])
        order = (0, 1, 2)
    elif patch == 4:
        sign = np.array([1, -1, 1])
        order = (1, 0, 2)
    elif patch == 5:
        sign = np.array([-1, 1, 1])
        order = (2, 1, 0)
    elif patch == 6:
        sign = np.array([1, 1, -1])
        order = (2, 1, 0)
    else:
        raise ValueError('patch must be an integer between 1 and 6, inclusive')

    return sign * xi_reference[..., order], sign * eta_reference[..., order]


def geom_weights(xi, eta, a=1., b=1., patch=None):
    """Calculate geometric weight at local coordinates

    Parameters
    ----------
    patch
    a
    b
    xi
    eta

    Returns
    -------

    """
    xi = np.array(xi)
    eta = np.array(eta)

    if a == 1 and b == 1:
        return (((1 + np.tan(xi)**2) * (1 + np.tan(eta)**2))
                / np.sqrt((1 + np.tan(xi)**2 + np.tan(eta)**2)**3))
    else:
        assert patch is not None

        xphi, ephi = phi_deriv(xi, eta, patch)

        ellipse_correction = np.array([a ** 2, a ** 2, b ** 2])
        E = np.sum(ellipse_correction * xphi ** 2, axis=-1)
        F = np.sum(ellipse_correction * xphi * ephi, axis=-1)
        G = np.sum(ellipse_correction * ephi ** 2, axis=-1)

        return np.sqrt(E * G - F ** 2)


def generate_grid(n_nodes, a=1., b=1.):
    """Generate a uniform mesh in patch coordinates xi and eta

    This function only generates the mesh for a single patch. The patch
    coordinate mesh is identical for each patch, the difference in
    patches is only taken into account in the mapping back to Cartesian
    coordinates.

    Parameters
    ----------
    a
    b
    n_nodes

    Returns
    -------

    """
    assert n_nodes % 2 == 0  # n_nodes should be even
    xi_mesh = np.linspace(-np.pi / 4, np.pi / 4, num=n_nodes + 1)
    eta_mesh = np.linspace(-np.pi / 4, np.pi / 4, num=n_nodes + 1)
    sphere_nodes = np.zeros(shape=(0, 3))
    ind_map = -np.ones(shape=(n_nodes+1, n_nodes+1, 6), dtype=int)
    for i in [0, 2, 1, 3, 4, 5]:  # Fill in opposite patches first
        patch = i + 1

        patch_nodes = phi(xi_mesh[:, np.newaxis], eta_mesh[np.newaxis, :],
                          patch=patch)

        patch_nodes = np.array(patch_nodes).transpose(
            (1, 2, 0))

        if patch == 1 or patch == 3:
            patch_nodes = patch_nodes.reshape((-1, 3))

            ind_map[..., i] = np.arange(
                sphere_nodes.shape[0], sphere_nodes.shape[0] + (n_nodes + 1)**2
            ).reshape((n_nodes+1, n_nodes+1))
        elif patch == 2 or patch == 4:
            patch_nodes = patch_nodes[1:-1].reshape((-1, 3))

            ind_map[1:-1, :, i] = np.arange(
                sphere_nodes.shape[0], sphere_nodes.shape[0] + (n_nodes**2 - 1)
            ).reshape((n_nodes-1, n_nodes+1))
            ind_map[0, :, i] = ind_map[-1, :, i-1]
            ind_map[-1, :, i] = ind_map[0, :, (i+1) % 4]
        elif patch == 5 or patch == 6:
            patch_nodes = patch_nodes[1:-1, 1:-1].reshape((-1, 3))

            ind_map[1:-1, 1:-1, i] = np.arange(
                sphere_nodes.shape[0], sphere_nodes.shape[0] + (n_nodes - 1)**2
            ).reshape((n_nodes-1, n_nodes-1))

            # A bunch of rules to ensure each edge matches up correctly
            if patch == 5:
                ind_map[0, :, i] = ind_map[::-1, -1, 3]
                ind_map[-1, :, i] = ind_map[:, -1, 1]
                ind_map[1:-1, 0, i] = ind_map[1:-1, -1, 0]
                ind_map[1:-1, -1, i] = ind_map[-2:0:-1, -1, 2]
            elif patch == 6:
                ind_map[0, :, i] = ind_map[:, 0, 3]
                ind_map[-1, :, i] = ind_map[::-1, 0, 1]
                ind_map[1:-1, 0, i] = ind_map[-2:0:-1, 0, 2]
                ind_map[1:-1, -1, i] = ind_map[1:-1, 0, 0]
        sphere_nodes = np.concatenate([sphere_nodes, patch_nodes])

    assert sphere_nodes.shape[0] == 6 * n_nodes ** 2 + 2
    assert np.count_nonzero(ind_map == -1) == 0

    nodes = np.array([b, a, a]) * sphere_nodes
    return eta_mesh, xi_mesh, nodes, ind_map


def stokeslet_integrand(x_tuple, center, eps, force):
    """Evaluate the regularized Stokeslet located at center

    This function returns the ith component of the free-space
    regularized Stokeslet centered at center, evaluated at position
    x_tuple.

    Parameters
    ----------
    force : callable or ndarray
        Strength of the regularized Stokeslet
    eps : float
        Blob parameter of the regularized Stokeslet
    center
    x_tuple : tuple of ndarray
        Tuple containing ndarrays of each dimension of the set of
        evaluation points

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
    try:
        f_array = force(x_array)
    except TypeError:
        f_array = force

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j, :] = np.dot(stokeslet[i, j, :, :],
                                     f_array[i, j, :])

    return -output / (8 * np.pi)


def h1(r, eps=0.01):
    sre = np.sqrt(r**2 + eps**2)
    return (8 * np.pi * sre) ** (-1) + eps**2 / (8 * np.pi * sre ** 3)


def h2(r, eps=0.01):
    return (8 * np.pi * np.sqrt(r**2 + eps**2) ** 3) ** (-1)


def d1(r, eps=0.01):
    sre = np.sqrt(r**2 + eps**2)
    return (4 * np.pi * sre ** 3) ** (-1) - 3 * eps**2 / (4 * np.pi * sre ** 5)


def d2(r, eps=0.01):
    sre = np.sqrt(r**2 + eps**2)
    return - 3 / (4 * np.pi * sre ** 5)


def h1p(r, eps=0.01):
    sre = np.sqrt(r**2 + eps**2)
    return -r / (8 * np.pi * sre**3) - 3 * r * eps**2 / (8 * np.pi * sre**5)


def h2p(r, eps=0.01):
    sre = np.sqrt(r**2 + eps**2)
    return -3 * r / (8 * np.pi * sre**5)


def wall_stokeslet_integrand(x_tuple, center, eps, force):
    """Evaluate the regularized Stokeslet located at center

    This function returns the ith component of the wall-bounded
    regularized Stokeslet centered at center, evaluated at position
    x_tuple.

    Parameters
    ----------
    x_tuple
    center
    eps
    force

    Returns
    -------

    """
    images = np.array([-1, 1, 1]) * center
    del_x = x_tuple - center
    im_x = x_tuple - images
    h = center.T[0]

    r = np.linalg.norm(del_x, axis=-1)
    r_im = np.linalg.norm(im_x, axis=-1)

    e_1 = np.array([1, 0, 0])
    dipole = 2 * np.dot(force, e_1) * e_1 - force
    rotlet = np.cross(force, e_1)

    term1 = force * h1(r, eps) + np.dot(force, del_x) * del_x * h2(r, eps)
    term2 = force * h1(r_im, eps) + np.dot(force, im_x) * im_x * h2(r_im, eps)
    term3 = dipole * d1(r_im, eps) + np.dot(dipole, im_x) * im_x * d2(r_im, eps)
    term4 = h1p(r_im, eps) / r_im + h2(r_im, eps)
    term5 = (np.dot(dipole, e_1) * im_x * h2(r_im, eps)
             + np.dot(im_x, e_1) * dipole * h2(r_im, eps)
             + np.dot(dipole, im_x) * e_1 * h1p(r_im, eps) / r_im
             + np.dot(im_x, e_1) * np.dot(dipole, im_x) * im_x
             * h2p(r_im, eps) / r_im)

    u = (term1 - term2 - h**2 * term3 + 2 * h * term4 * np.cross(rotlet, im_x)
         + 2 * h * term5)
    return u


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
            sphere_integrate(stokeslet_integrand, n_nodes=n_nodes, center=point, eps=eps, force=point_force)
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


def sphere_integrate(integrand, n_nodes=16, a=1., b=1., **kwargs):
    """Integrates a function over the surface of a spheroid

    Parameters
    ----------
    integrand
    n_nodes
    a
    b
    kwargs

    Returns
    -------

    """
    if type(integrand) is np.ndarray:
        assert integrand.shape[0] == 6*n_nodes**2 + 2
    eta_mesh, xi_mesh, cart_nodes, ind_map = generate_grid(n_nodes, a=a, b=b)

    del_xi = xi_mesh[1] - xi_mesh[0]
    del_eta = eta_mesh[1] - eta_mesh[0]

    patch_integrals = list()
    for i in range(6):
        patch = i + 1

        g_matrix = geom_weights(xi_mesh[:, np.newaxis],
                                eta_mesh[np.newaxis, :], a=a, b=b, patch=patch)

        if type(integrand) is np.ndarray:
            f_matrix = integrand[ind_map[..., i]]
        else:
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
