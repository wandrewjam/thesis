import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
from ce_convergence_test import spheroid_surface_area


def main():
    a, b = 1.5, 0.5

    with open('ho_errors.pkl', 'r') as f:
        err = pickle.load(f)

    with open('ho_convergence.pkl', 'r') as f:
        conv = pickle.load(f)

    line_fmt = '{} & {} & {} & {} \\\\'

    distance_ratios = np.array([10.0677, 3.7622, 1.5431,
                                1.1276, 1.0453])
    d_list = distance_ratios * a
    surf_area = spheroid_surface_area(a, b)
    d_dict = {d_list[0]: '#c7e9c0', d_list[1]: '#a1d99b', d_list[2]: '#74c476',
              d_list[3]: '#31a354', d_list[4]: '#006d2c'}
    theta_dict = {0: '-', np.pi / 2: '--', np.pi: '-.'}
    phi_dict = {0: 'o', np.pi / 4: 'v', np.pi / 2: '*'}
    fig, ax = plt.subplots()
    for key, err_sequence in err.items():
        exact = conv[key][36]
        max_el = np.amax(np.concatenate(exact, axis=1))

        sorter = np.argsort(err_sequence.keys())
        mesh_size = np.array(err_sequence.keys())
        errors = np.array(err_sequence.values()) / max_el

        N_nodes = 6 * mesh_size ** 2 + 2
        diameter = np.sqrt(surf_area / N_nodes)

        if key[0] == 0. and key[1] == 0.:
            print('\n d = {} \n'.format(key[2]))
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

        ax.plot(diameter[sorter], errors[sorter], c=d_dict[key[2]],
                linestyle=theta_dict[key[1]], marker=phi_dict[key[0]])

    d_fmt = '$d = {}$'
    handles = [
        mlines.Line2D([], [], color='#c7e9c0', label=d_fmt.format(d_list[0])),
        mlines.Line2D([], [], color='#a1d99b', label=d_fmt.format(d_list[1])),
        mlines.Line2D([], [], color='#74c476', label=d_fmt.format(d_list[2])),
        mlines.Line2D([], [], color='#31a354', label=d_fmt.format(d_list[3])),
        mlines.Line2D([], [], color='#006d2c', label=d_fmt.format(d_list[4])),
        mlines.Line2D([], [], color='k', linestyle='-', label='$\\theta = 0.$'),
        mlines.Line2D([], [], color='k', linestyle='--', label='$\\theta = \\pi/2$'),
        mlines.Line2D([], [], color='k', linestyle='-.', label='$\\theta = \\pi$'),
        mlines.Line2D([], [], color='k', linestyle='-', marker='o', label='$\\phi = 0$'),
        mlines.Line2D([], [], color='k', linestyle='-', marker='v', label='$\\phi = \\pi/4$'),
        mlines.Line2D([], [], color='k', linestyle='-', marker='*', label='$\\phi = \\pi/2$'),
    ]
    ax.legend(handles=handles)
    ax.set_xlabel('Discretization size $h$')
    ax.set_ylabel('Relative error')
    plt.show()


if __name__ == '__main__':
    main()
