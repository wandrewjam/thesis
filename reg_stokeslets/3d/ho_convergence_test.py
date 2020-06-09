import matplotlib.pyplot as plt
import pickle
from resistance_matrix_test import generate_resistance_matrices
from sphere_integration_utils import compute_helper_funs
import cProfile
import line_profiler


def spheroid_surface_area(a, b):
    assert a > b
    e = np.sqrt(1 - b**2 / a**2)
    return 2 * np.pi * a ** 2 * (1 + b ** 2 * np.arctanh(e) / (e * a ** 2))


def main(server='mac', proc=1):
    a, b = 1.5, 0.5
    surf_area = spheroid_surface_area(a, b)

    c, n = .6, 1.0

    distance_ratios = np.array([10.0677, 3.7622, 1.5431,
                                1.1276, 1.0453])
    phi_list = np.linspace(0, np.pi / 2, num=3)
    theta_list = np.linspace(0, np.pi, num=3)
    d_list = distance_ratios * a

    if server == 'linux':
        n_nodes = (1 + np.arange(9)) * 4
    elif server == 'mac':
        n_nodes = (1 + np.arange(4)) * 4
    else:
        raise ValueError('server is invalid')

    n_max = np.amax(n_nodes)

    result_dict = {}
    exact_dict = {}
    h = np.sqrt(surf_area / (6 * n_nodes ** 2 + 2))
    epsilon = c * h ** n

    # Precompute up to r=25
    precompute_arrays = list()
    r_save = np.linspace(0, 34, num=34 * 10**3)
    for ii in range(len(epsilon)):
        e = epsilon[ii]
        precompute_array = (r_save, ) + tuple(compute_helper_funs(r_save, e))
        precompute_arrays.append(precompute_array)

    for phi in phi_list:
        for theta in theta_list:
            for d in d_list:
                result = [
                    (node,
                     generate_resistance_matrices(
                         e, node, a=a, b=b, domain='wall', distance=d,
                         theta=theta, phi=phi, shear_vec=True, proc=proc,
                         precompute_array=precompute_array
                     )) for (e, node, precompute_array)
                    in zip(epsilon, n_nodes, precompute_arrays)
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
                result_dict.update([((phi, theta, d), processed_result)])
                exact_dict.update([((phi, theta, d), processed_result[n_max])])

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
    import os
    import sys

    server = sys.argv[1]
    num_threads = sys.argv[2]
    try:
        arg3 = sys.argv[3]
        if arg3 == 'profile':
            profile = True
        else:
            profile = False
    except IndexError:
        profile = False

    if server == 'linux':
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    elif server == 'mac':
        os.environ["MKL_NUM_THREADS"] = num_threads
    import numpy as np

    if profile:
        # Check that the stats file doesn't already exist
        stats_file = 'ho_sym_lookup_conv_profile.stats'

        if os.path.isfile(stats_file):
            raise NameError('file already exists!')
        else:
            cProfile.run('main(server=server, proc=int(num_threads))',
                         filename=stats_file)
    else:
        main(server=server, proc=int(num_threads))
