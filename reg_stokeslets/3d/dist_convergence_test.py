import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import matplotlib.pyplot as plt
import pickle
from resistance_matrix_test import generate_resistance_matrices


def spheroid_surface_area(a, b):
    assert a > b
    e = np.sqrt(1 - b**2 / a**2)
    return 2 * np.pi * a ** 2 * (1 + b ** 2 * np.arctanh(e) / (e * a ** 2))


def main(server='mac'):
    a, b = 1.5, 0.5
    surf_area = spheroid_surface_area(a, b)

    c, n = 1.0, 1.0

    d_list = np.array([1.5, 1.0, .75, .6, .51, .501])

    if server == 'linux':
        n_nodes = (1 + np.arange(9)) * 4
    elif server == 'mac':
        n_nodes = (1 + np.arange(4)) * 4
    else:
        raise ValueError('server is invalid')

    n_max = np.amax(n_nodes)

    result_dict = {}
    exact_dict = {}
    for d in d_list:
        h = np.sqrt(surf_area / (6 * n_nodes ** 2 + 2))
        epsilon = c * h ** n
        result = [
            (node,
             generate_resistance_matrices(
                 e, node, a=a, b=b, domain='wall', shear_vec=True,
                 distance=d)
             )
            for (e, node) in zip(epsilon, n_nodes)
        ]

        processed_result = list()
        for node, matrices in result:
            t, p, pt, r, s1, s2 = matrices
            resistance_matrix = np.block([[t, p], [pt, r]])
            shear_vec = np.concatenate([s1, s2], axis=0)
            assert resistance_matrix.shape == (6, 6)
            assert shear_vec.shape == (6, 1)

            processed_result.append((node, (resistance_matrix, shear_vec)))

        processed_result = dict(processed_result)
        result_dict.update([(d, processed_result)])
        exact_dict.update([(d, processed_result[n_max])])

    with open('ho_convergence.pkl', 'w') as f:
        pickle.dump(result_dict, f)

    errors = {}
    for key, value in result_dict.items():
        exact_matrix, exact_shear = value[n_max]
        err_sequence = list()
        for node, matrices in value.items():
            error = np.amax(np.concatenate(
                [matrices[0] - exact_matrix, matrices[1] - exact_shear], axis=1
            ))
            err_sequence.append((node, error))

        errors.update([(key, dict(err_sequence))])

    with open('ho_errors.pkl', 'w') as f:
        pickle.dump(errors, f)

    if server == 'mac':
        fig, ax = plt.subplots()
        for key, err in errors.items():
            sorter = np.argsort(err.keys())
            jonathan, vals = np.array(err.keys()), np.array(err.values())
            jonathan_h = np.sqrt(surf_area / (6 * jonathan ** 2 + 2))
            label_str = 'd = {}, $\\theta$ = {}, $\\phi$ = {}'.format(
                key[2], key[1], key[0])
            ax.plot(jonathan_h[sorter], vals[sorter], label=label_str)
        ax.legend()
        ax.set_xlabel('Discretization size ($h$)')
        ax.set_ylabel('Relative Error')
        plt.show()

    print()


if __name__ == '__main__':
    import sys
    main(server=sys.argv[1])
