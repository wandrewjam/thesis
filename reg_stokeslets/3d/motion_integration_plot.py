import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
from motion_integration import find_min_separation, evaluate_motion_equations


def parse_info(filename):
    info_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            key, val_str = line[:-1].split(', ', 1)
            if key in ['distance', 'stop', 'coarse time', 'fine time']:
                val = float(val_str)
            elif key in ['steps', 'coarse counter', 'fine counter']:
                val = int(val_str)
            elif key == 'order':
                val = val_str
            elif key == 'exact solution' or key == 'adaptive':
                val = val_str == 'True'
            elif key == 'e0':
                val = tuple([float(s) for s in val_str[1:-1].split(', ')])
            info_dict.update([(key, val)])

    return info_dict


def find_separation(x1, e1, e2, e3):
    sep = np.zeros(shape=x1.shape)
    e_m = np.stack([e1, e2, e3])
    for i in range(len(x1)):
        sep[i] = find_min_separation(x1[i], e_m[:, i])
    return sep


def main(file_suffix, save_plots=False):
    # Define directories
    import os
    plot_dir = os.path.expanduser('~/thesis/meeting-notes/summer-20/'
                                  'notes_081220/')
    data_dir = 'data/'

    # Load data
    fine_data = np.load(data_dir + 'fine' + file_suffix + '.npz')
    coarse_data = np.load(data_dir + 'coarse' + file_suffix + '.npz')

    info = parse_info(data_dir + 'info' + file_suffix + '.txt')

    # Define local variables
    t, errs = coarse_data['t'], coarse_data['errs']
    x1, x2, x3 = coarse_data['x1'], coarse_data['x2'], coarse_data['x3']
    e1, e2, e3 = coarse_data['e1'], coarse_data['e2'], coarse_data['e3']

    x1_fine, x2_fine, x3_fine = (fine_data['x1'], fine_data['x2'],
                                 fine_data['x3'])
    e1_fine, e2_fine, e3_fine = (fine_data['e1'], fine_data['e2'],
                                 fine_data['e3'])

    if file_suffix == '72':
        bad_data = np.load(data_dir + 'coarse18.npz')
        x1c, x2c, x3c = bad_data['x1'], bad_data['x2'], bad_data['x3']
        e1c, e2c, e3c = bad_data['e1'], bad_data['e2'], bad_data['e3']
    elif file_suffix == '73':
        bad_data = np.load(data_dir + 'coarse19.npz')
        x1c, x2c, x3c = bad_data['x1'], bad_data['x2'], bad_data['x3']
        e1c, e2c, e3c = bad_data['e1'], bad_data['e2'], bad_data['e3']

    try:
        n_array = coarse_data['node_array']
        sep_array = coarse_data['sep_array']
    except KeyError:
        # Backwards compatibility with old code w/o node array saved
        pass

    exact_solution = info['exact solution']
    try:
        adaptive = info['adaptive']
    except KeyError:
        # Backwards compatibility with old code w/o adaptive specified
        adaptive = False
    order = info['order']

    fine_sep_array = find_separation(x1_fine, e1_fine, e2_fine, e3_fine)
    coarse_sep_array = find_separation(x1c, e1c, e2c, e3c)

    # Plot numerical and analytical solutions
    ax1 = host_subplot(111)
    ax_tw = ax1.twinx()

    ax1.set_xlabel('Time elapsed')
    ax1.set_ylabel('$x$ component of CoM')
    # ax_tw.tick_params(axis='y')
    ax_tw.set_ylabel('$y$ and $z$ components of CoM')

    # Pick the curve labels
    if exact_solution:
        x1_label, x1f_label, x2_label, x2f_label, x3_label, x3f_label = (
            '$x$ approx', '$x$ exact', '$y$ approx', '$y$ exact',
            '$z$ approx', '$z$ exact')
    elif adaptive:
        x1_label, x1f_label, x2_label, x2f_label, x3_label, x3f_label = (
            '$x$ adapt', '$x$ fine', '$y$ adapt', '$y$ fine',
            '$z$ adapt', '$z$ fine')
    else:
        x1_label, x1f_label, x2_label, x2f_label, x3_label, x3f_label = (
            '$x$ coarse', '$x$ fine', '$y$ coarse', '$y$ fine',
            '$z$ coarse', '$z$ fine')

    # Plot the 6 curves
    ax1.plot(t, x1, label=x1_label, color='tab:blue')
    ax1.plot(t, x1_fine, label=x1f_label, color='tab:orange')
    ax_tw.plot(t, x2, label=x2_label, color='tab:red')
    ax_tw.plot(t, x2_fine,label=x2f_label, color='tab:purple')
    ax_tw.plot(t, x3, label=x3_label, color='tab:pink')
    ax_tw.plot(t, x3_fine, label=x3f_label, color='tab:gray')
    if file_suffix == '72' or file_suffix == '73':
        ax1.plot(t, x1c, label='$x$ coarse', color='tab:green')
        ax_tw.plot(t, x2c, label='$y$ coarse', color='tab:brown')
        ax_tw.plot(t, x3c, label='$z$ coarse', color='tab:olive')

    ax1.legend()

    if save_plots:
        plt.savefig(plot_dir + 'com_plot{}_{}'.format(file_suffix, order),
                    bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    fig2, ax2 = plt.subplots()

    # Pick the curve labels
    if exact_solution:
        e1_label, e2_label, e3_label, e1f_label, e2f_label, e3f_label = (
            '$e_{mx}$ approx', '$e_{my}$ approx', '$e_{mz}$ approx',
            '$e_{mx}$ exact', '$e_{my}$ exact', '$e_{mz}$ exact')
    elif adaptive:
        e1_label, e2_label, e3_label, e1f_label, e2f_label, e3f_label = (
            '$e_{mx}$ adapt', '$e_{my}$ adapt', '$e_{mz}$ adapt',
            '$e_{mx}$ fine', '$e_{my}$ fine', '$e_{mz}$ fine')
    else:
        e1_label, e2_label, e3_label, e1f_label, e2f_label, e3f_label = (
            '$e_{mx}$ coarse', '$e_{my}$ coarse', '$e_{mz}$ coarse',
            '$e_{mx}$ fine', '$e_{my}$ fine', '$e_{mz}$ fine')
    ax2.set_xlabel('Time elapsed')
    ax2.set_ylabel('Orientation components')

    # Plot the 6 curves
    ax2.plot(t, e1, label=e1_label, color='tab:blue')
    ax2.plot(t, e1_fine, label=e1f_label, color='tab:orange')
    ax2.plot(t, e2, label=e2_label, color='tab:red')
    ax2.plot(t, e2_fine, label=e2f_label, color='tab:purple')
    ax2.plot(t, e3, label=e3_label, color='tab:pink')
    ax2.plot(t, e3_fine, label=e3f_label, color='tab:gray')
    if file_suffix == '72' or file_suffix == '73':
        ax2.plot(t, e1c, label='$e_{mx}$ coarse', color='tab:green')
        ax2.plot(t, e2c, label='$e_{my}$ coarse', color='tab:brown')
        ax2.plot(t, e3c, label='$e_{mz}$ coarse', color='tab:olive')

    ax2.legend()

    if save_plots:
        fig2.savefig(plot_dir + 'orient_plot{}_{}'.format(file_suffix, order),
                     bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    fig3, ax3 = plt.subplots()
    ax3.plot(t, e1 - e1_fine, t, e2 - e2_fine, t, e3 - e3_fine)
    ax3.legend(['$e_{mx}$ error', '$e_{my}$ error', '$e_{mz}$ error'])
    ax3.set_xlabel('Time elapsed')
    if exact_solution:
        ax3.set_ylabel('Absolute error (approx - exact)')
    else:
        ax3.set_ylabel('Estimated error (coarse - fine)')
    if save_plots:
        fig3.savefig(plot_dir + 'orient_err_plot{}_{}'
                     .format(file_suffix, order), bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()

    # fig_mag, ax_mag = plt.subplots()
    # ax_mag.plot(t, np.linalg.norm(e_m, axis=0),
    #             t, np.sqrt(e1_exact**2 + e2_exact**2 + e3_exact**2))
    # ax_mag.legend(['$e_m$ magnitude', 'Exact magnitude'])
    # ax_mag.set_xlabel('Time elapsed')
    # ax_mag.set_ylabel('Magnitude')
    # if not save_plots:
    #     plt.tight_layout()
    #     plt.show()

    if exact_solution:
        fig4, ax4 = plt.subplots()
        ax4.plot(t[1:], errs[:, 1:].T)
        ax4.set_xlabel('Time elapsed')
        ax4.set_ylabel('Velocity error (approx - exact)')
        ax4.legend(['$v_x$ error', '$v_y$ error', '$v_z$ error',
                    '$\\omega_x$ error', '$\\omega_y$ error',
                    '$\\omega_z$ error'])
        if save_plots:
            fig4.savefig(plot_dir + 'vel_err_plot{}_{}'
                         .format(file_suffix, order), bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()

    try:
        ax_n = host_subplot(111)
        ax_sep = ax_n.twinx()
        ax_n.plot(t[:-1], n_array[:-1], label='$N$ choice', color='k')
        ax_sep.plot(t[:-1], sep_array[:-1], label='Adaptive sep',
                    color='tab:blue')
        ax_sep.plot(t[:-1], fine_sep_array[:-1], label='Fine sep',
                    color='tab:orange')
        ax_sep.plot(t[:-1], coarse_sep_array[:-1], label='Coarse sep',
                    color='tab:green')
        ax_n.set_xlabel('Time elapsed')
        ax_n.set_ylabel('N chosen')
        ax_sep.set_ylabel('Plt-wall separation')
        ax_n.legend()

        if save_plots:
            plt.savefig(plot_dir + 'nsep_plot{}'.format(file_suffix),
                        bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
    except NameError:
        pass

    print('Done!')


if __name__ == '__main__':
    import sys

    suffix = sys.argv[1]

    try:
        save = sys.argv[2]
        save_bool = save == 'T'
        main(suffix, save_bool)
    except IndexError:
        main(suffix)
