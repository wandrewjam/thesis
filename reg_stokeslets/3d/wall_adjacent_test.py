from __future__ import division
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
from resistance_matrix_test import generate_resistance_matrices


def main(a=1., b=1., theta=0., phi=0., server='mac'):
    """Replicate convergence tests of Ainley et. al., 2008

    Parameters
    ----------
    theta
    phi
    a
    b
    server

    Returns
    -------

    """
    true_par = np.array([1., 1.0591, 1.1738, 1.5675, 2.1515, 2.6475, 3.7863])
    true_prp = np.array([1., 1.1253, 1.4129, 3.0361, 9.2518, 23.6605, 201.864])
    true_srf = np.array([1., 1.0587, 1.1671, 1.4391, 1.6160, 1.6682, 1.6969])
    true_srt = np.array([1., 0.9998, 0.9997, 0.9742, 0.9537, 0.9477, 0.9444])

    if server == 'linux':
        n_nodes = np.array([8, 12, 16, 20])
        h = np.sqrt((4 * np.pi * a**2) / (6 * n_nodes**2 + 2))
        eps = 0.22 * h**0.9  # From Ainley et. al., 2008

        dist_ratios = np.array([11013.2, 10.0677, 3.7622, 1.5431,
                                1.1276, 1.0453, 1.005])
    elif server == 'mac':
        n_nodes = np.array([8, 12])
        h = np.sqrt((4 * np.pi * a**2) / (6 * n_nodes**2 + 2))
        eps = 0.22 * h**0.9

        dist_ratios = np.array([11013.2, 10.0677, 3.7622])
    else:
        raise ValueError('server is not valid')

    distances = dist_ratios * a

    result = [[
        generate_resistance_matrices(e, n, a=a, b=b, domain='wall', distance=distance)
        for distance in distances] for (e, n) in zip(eps, n_nodes)
    ]

    par_table = np.zeros(shape=(len(dist_ratios), 3 + len(n_nodes)))
    prp_table = np.zeros(shape=(len(dist_ratios), 3 + len(n_nodes)))
    srf_table = np.zeros(shape=(len(dist_ratios), 3 + len(n_nodes)))
    srt_table = np.zeros(shape=(len(dist_ratios), 3 + len(n_nodes)))

    par_table[:, 0] = dist_ratios
    par_table[:, 1] = distances - a
    par_table[:, -1] = true_par[:len(dist_ratios)]
    prp_table[:, 0] = dist_ratios
    prp_table[:, 1] = distances - a
    prp_table[:, -1] = true_prp[:len(dist_ratios)]
    srf_table[:, 0] = dist_ratios
    srf_table[:, 1] = distances - a
    srf_table[:, -1] = true_srf[:len(dist_ratios)]
    srt_table[:, 0] = dist_ratios
    srt_table[:, 1] = distances - a
    srt_table[:, -1] = true_srt[:len(dist_ratios)]

    for (i, dist) in enumerate(distances):
        for (j, n) in enumerate(n_nodes):
            t_matrix = result[j][i][0]
            shear_f = result[j][i][4]
            shear_t = result[j][i][5]
            par_table[i, j + 2] = t_matrix[1, 1] / (6 * np.pi * a)
            prp_table[i, j + 2] = t_matrix[0, 0] / (6 * np.pi * a)
            srf_table[i, j + 2] = shear_f[2] / (6 * np.pi * a * dist)
            srt_table[i, j + 2] = shear_t[1] / (4 * np.pi * a**3)

    header = 'd/a, Gap size, '
    header += '{}, ' * len(n_nodes)
    header += 'F*'
    header = header.format(*[6 * n**2 + 2 for n in n_nodes])

    np.savetxt('par_table.dat', par_table, delimiter=',', header=header)
    np.savetxt('prp_table.dat', prp_table, delimiter=',', header=header)
    np.savetxt('srf_table.dat', srf_table, delimiter=',', header=header)
    np.savetxt('srt_table.dat', srt_table, delimiter=',', header=header)


if __name__ == '__main__':
    import sys
    kwargs = {
        'a': float(sys.argv[1]),
        'b': float(sys.argv[2]),
        'theta': float(sys.argv[3]),
        'phi': float(sys.argv[4]),
        'server': sys.argv[5],
    }

    main(**kwargs)
