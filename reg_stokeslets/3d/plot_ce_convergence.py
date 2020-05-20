import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
from ce_convergence_test import spheroid_surface_area


def main():
    a, b = 1.5, 0.5
    with open('ce_convergence.pkl', 'r') as f:
        conv = pickle.load(f)

    with open('ce_errors.pkl', 'r') as f:
        err = pickle.load(f)

    surf_area = spheroid_surface_area(a, b)
    c_dict = {0.6: '#bae4b3', 0.8: '#74c476', 1.: '#31a354'}
    l_dict = {0.8: '-', 0.9: '--', 1.0: '-.'}

    line_fmt = '{} & {} & {} & {} \\\\'
    fig, ax = plt.subplots()
    for key, err_sequence in err.items():
        exact = conv[key][36]
        max_el = np.amax(np.concatenate(exact, axis=1))

        sorter = np.argsort(err_sequence.keys())
        mesh_size = np.array(err_sequence.keys())
        errors = np.array(err_sequence.values()) / max_el

        N_nodes = 6 * mesh_size ** 2 + 2
        diameter = np.sqrt(surf_area / N_nodes)

        if key[0] == 0.6:
            print('\n C = 0.6, n = {} \n'.format(key[1]))
            table = ('$h$ & $N$ & error & $p \\approx \\frac{\\log(e_{i+1} / '
                     'e_i)}{\\log(h_{i+1} / h_i)}$ \\\\')
            print(table)
            for i in range(len(errors) - 1):
                h = diameter[sorter][i]
                N = N_nodes[sorter][i]
                error = errors[sorter][i]
                p = (np.log(errors[sorter][i+1] / errors[sorter][i])
                     / np.log(diameter[sorter][i+1] / diameter[sorter][i]))
                lines = line_fmt.format(h, N, error, p)
                print(lines)

        ax.plot(diameter[sorter], errors[sorter], c=c_dict[key[0]], marker='.',
                linestyle=l_dict[key[1]])
    handles = [
        mlines.Line2D([], [], color='#74c476', label='$c = 0.6$'),
        mlines.Line2D([], [], color='#31a354', label='$c = 0.8$'),
        mlines.Line2D([], [], color='#006d2c', label='$c = 1.0$'),
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
