import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
from ce_convergence_test import spheroid_surface_area


def main():
    a, b = 1.5, 0.5

    with open('ce_errors.pkl', 'r') as f:
        err = pickle.load(f)

    surf_area = spheroid_surface_area(a, b)
    c_dict = {0.1: '#bae4b3', 0.2: '#74c476', 0.4: '#31a354', 0.6: '#006d2c'}
    l_dict = {0.8: '-', 0.9: '--', 1.0: '-.'}
    fig, ax = plt.subplots()
    for key, err_sequence in err.items():
        sorter = np.argsort(err_sequence.keys())
        mesh_size = np.array(err_sequence.keys())
        errors = np.array(err_sequence.values())

        diameter = np.sqrt(surf_area / (6 * mesh_size ** 2 + 2))

        ax.plot(diameter[sorter], errors[sorter], c=c_dict[key[0]],
                linestyle=l_dict[key[1]])
    handles = [
        mlines.Line2D([], [], color='#bae4b3', label='$c = 0.1$'),
        mlines.Line2D([], [], color='#74c476', label='$c = 0.2$'),
        mlines.Line2D([], [], color='#31a354', label='$c = 0.4$'),
        mlines.Line2D([], [], color='#006d2c', label='$c = 0.6$'),
        mlines.Line2D([], [], color='k', linestyle='-', label='$n = 0.8$'),
        mlines.Line2D([], [], color='k', linestyle='--', label='$n = 0.9$'),
        mlines.Line2D([], [], color='k', linestyle='-.', label='$n = 1.0$'),
    ]
    ax.legend(handles=handles)
    ax.set_xlabel('Discretization size $h$')
    ax.set_ylabel('Relative error')
    plt.show()


if __name__ == '__main__':
    main()
