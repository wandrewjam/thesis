import numpy as np

# This script computes the exact resistance matrices for an oblate
# spheroid in an unbounded domain


def eccentricity(a, c):
    """Compute the eccentricity of a spheroid

    Parameters
    ----------
    a
    c

    Returns
    -------

    """
    assert c > 0
    assert a >= c

    e = np.sqrt(1 - (float(c) / a) ** 2)

    assert 0 < e < 1
    return e


def x_a(e):
    """Compute 1st translational resistance function for spheroids

    Parameters
    ----------
    e

    Returns
    -------

    """
    assert 0 < e < 1

    sqr = np.sqrt(1 - e**2)
    denom = 2 * (2 * e**2 - 1) * np.arctan(e / sqr) + 2 * e * sqr

    return (8. / 3) * e**3 / denom


def y_a(e):
    """Compute 2nd translational resistance function for spheroids

    Parameters
    ----------
    e

    Returns
    -------

    """
    assert 0 < e < 1

    sqr = np.sqrt(1 - e**2)
    denom = (2 * e**2 + 1) * np.arctan(e / sqr) - e * sqr
    return (8. / 3) * e**3 / denom


def x_c(e):
    """Compute 1st rotational resistance function for spheroids

    Parameters
    ----------
    e

    Returns
    -------

    """
    assert 0 < e < 1

    sqr = np.sqrt(1 - e**2)
    denom = np.arctan(e / sqr) - e * sqr
    return (2. / 3) * e**3 / denom


def y_c(e):
    """Compute 2nd rotational resistance function for spheroids

    Parameters
    ----------
    e

    Returns
    -------

    """
    assert 0 < e < 1

    sqr = np.sqrt(1 - e**2)
    denom = e * sqr - (1 - 2 * e**2) * np.arctan(e / sqr)
    return (2. / 3) * e**3 * (2 - e**2) / denom


def y_h(e):
    """Compute rate-of-strain resistance function for spheroids

    Parameters
    ----------
    e

    Returns
    -------

    """
    assert 0 < e < 1

    sqr = np.sqrt(1 - e**2)
    denom = e * sqr - (1 - 2 * e**2) * np.arctan(e / sqr)
    return (2. / 3) * e**5 / denom


def t_matrix(a, c):
    """Compute the translational resistance tensor for spheroids

    Parameters
    ----------
    a
    c

    Returns
    -------

    """
    assert c > 0
    assert a >= c

    e = eccentricity(a, c)
    prp = y_a(e)
    par = x_a(e)
    t = np.diag([prp, prp, par])
    return 6 * np.pi * a * t


def r_matrix(a, c):
    """Compute the rotational resistance tensor for spheroids

    Parameters
    ----------
    a
    c

    Returns
    -------

    """
    assert c > 0
    assert a >= c

    e = eccentricity(a, c)
    prp = y_c(e)
    par = x_c(e)
    r = np.diag([prp, prp, par])
    return 8 * np.pi * a**3 * r


def s_vector(a, c, orientation):
    """Compute the shear resistance vector for spheroids

    Parameters
    ----------
    orientation
    a
    c

    Returns
    -------

    """
    assert c > 0
    assert a >= c

    # Compute the rate-of-strain part
    e = eccentricity(a, c)
    res = y_h(e)
    ee = np.zeros(shape=(3, 3))
    ee[1, 2] = 1.
    ee[2, 1] = 1.
    ee /= 2

    perm_tensor = np.zeros(shape=(3, 3, 3))
    perm_tensor[0, 1, 2] = 1.
    perm_tensor[1, 0, 2] = -1.
    perm_tensor[1, 2, 0] = 1.
    perm_tensor[2, 1, 0] = -1.
    perm_tensor[2, 0, 1] = 1.
    perm_tensor[0, 2, 1] = -1.

    tmp1 = perm_tensor[:, :, np.newaxis, :] * orientation[:, np.newaxis]
    tmp2 = perm_tensor[:, np.newaxis, :, :] * orientation[:, np.newaxis,
                                                          np.newaxis]
    tmp3 = tmp1 + tmp2
    tmp4 = np.dot(tmp3, orientation)
    s = np.tensordot(tmp4, ee, axes=2)

    # Compute the rotational part
    prp = x_c(e)
    par = y_c(e)
    om_inf = np.array([-.5, 0, 0])
    rot = np.dot(prp * np.outer(orientation, orientation)
                 + par * (np.eye(3) - np.outer(orientation, orientation)),
                 om_inf)

    return 8 * np.pi * a**3 * (rot + res * s / 2)


if __name__ == '__main__':
    a, c = 1.5, .5

    print(t_matrix(a, c))
    print()
    print(r_matrix(a, c))
    print()
    print(s_vector(a, c, np.array([0, 0, 1.])))
