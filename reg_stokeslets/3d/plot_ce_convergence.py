import numpy as np
import matplotlib.pyplot as plt
import pickle
from ce_convergence_test import spheroid_surface_area


def main():
    a, b = 1.5, 0.5

    with open('ce_errors.pkl', 'r') as f:
        err = pickle.load(f)

    surf_area = spheroid_surface_area(a, b)
    c_dict = {0.1: 'b', 0.2: 'k', 0.4: 'r', 0.6: 'g'}
    l_dict = {0.8: '-', 0.9: '--', 1.0: '-.'}
    fig, ax = plt.subplots()
    for key, err_sequence in err.items():
        sorter = np.argsort(err_sequence.keys())
        mesh_size = np.array(err_sequence.keys())
        errors = np.array(err_sequence.values())

        diameter = np.sqrt(surf_area / (6 * mesh_size ** 2 + 2))

        label_str = 'c = {}, n = {}'.format(key[0], key[1])
        ax.plot(diameter[sorter], errors[sorter], c=c_dict[key[0]],
                linestyle=l_dict[key[1]], label=label_str)
    ax.legend()
    ax.set_xlabel('Discretization size $h$')
    ax.set_ylabel('Relative error')
    plt.show()


if __name__ == '__main__':
    main()
