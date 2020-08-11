from __future__ import division
import numpy as np
from resistance_matrix_test import generate_resistance_matrices


def main(a=1., b=1., theta=0., phi=0., server='mac', expt='par'):
    """Replicate convergence tests of Ainley et. al., 2008

    Parameters
    ----------
    expt
    theta
    phi
    a
    b
    server

    Returns
    -------

    """
    if expt == 'par':
        true = np.array([1., 1.0591, 1.1738, 1.5675,
                             2.1515, 2.6475, 3.7863])
        C = 0.22  # From Ainley et. al., 2008
    elif expt == 'prp':
        true = np.array([1., 1.1253, 1.4129, 3.0361,
                             9.2518, 23.6605, 201.864])
        C = 0.4  # From Ainley et. al., 2008
    elif expt == 'srf':
        true = np.array([1., 1.0587, 1.1671, 1.4391,
                             1.6160, 1.6682, 1.6969])
        C = 0.22
    elif expt == 'srt':
        true = np.array([1., 0.9998, 0.9997, 0.9742,
                             0.9537, 0.9477, 0.9444])
        C = 0.22
    else:
        raise ValueError('expt is not valid')

    if server == 'linux':
        n_nodes = np.array([8, 12, 16, 20])
        h = np.sqrt((4 * np.pi * a**2) / (6 * n_nodes**2 + 2))
        eps = C * h ** 0.9  # From Ainley et. al., 2008

        dist_ratios = np.array([11013.2, 10.0677, 3.7622, 1.5431,
                                1.1276, 1.0453, 1.005])
    elif server == 'mac':
        n_nodes = np.array([8, 12])
        h = np.sqrt((4 * np.pi * a**2) / (6 * n_nodes**2 + 2))
        eps = C * h ** 0.9

        dist_ratios = np.array([11013.2, 10.0677, 3.7622])
    else:
        raise ValueError('server is not valid')

    distances = dist_ratios * a

    result = [[
        generate_resistance_matrices(e, n, a=a, b=b, domain='wall', distance=distance, proc=16)
        for distance in distances] for (e, n) in zip(eps, n_nodes)
    ]

    table = np.zeros(shape=(len(dist_ratios), 3 + len(n_nodes)))
    table[:, 0] = dist_ratios
    table[:, 1] = distances - a
    table[:, -1] = true[:len(dist_ratios)]

    for (i, dist) in enumerate(distances):
        for (j, n) in enumerate(n_nodes):
            t_matrix = result[j][i][0]
            shear_f = result[j][i][4]
            shear_t = result[j][i][5]

            if expt == 'par':
                table[i, j + 2] = t_matrix[1, 1] / (6 * np.pi * a)
            elif expt == 'prp':
                table[i, j + 2] = t_matrix[0, 0] / (6 * np.pi * a)
            elif expt == 'srf':
                table[i, j + 2] = shear_f[2] / (6 * np.pi * a * dist)
            elif expt == 'srt':
                table[i, j + 2] = shear_t[1] / (4 * np.pi * a**3)
            else:
                raise ValueError('expt is not valid')

    header = 'd/a, Gap size, '
    header += '{}, ' * len(n_nodes)
    header += 'F*'
    header = header.format(*[6 * n**2 + 2 for n in n_nodes])

    np.savetxt(expt + '_table.dat', table, delimiter=',', header=header)


if __name__ == '__main__':
    import sys
    kwargs = {
        'a': float(sys.argv[1]),
        'b': float(sys.argv[2]),
        'theta': float(sys.argv[3]),
        'phi': float(sys.argv[4]),
        'server': sys.argv[5],
        'expt': sys.argv[6]
    }

    main(**kwargs)
