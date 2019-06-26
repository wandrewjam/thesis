import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


def main(filename, show_plot=False):
    npz_dir = '../npz-files/'
    plot_dir = '../plots/'
    data = np.load(npz_dir + filename + '.npz')
    y, s_eval = data['y'], data['s_eval']
    u1_bdy = data['u1_bdy']

    s_mask = s_eval > 2. / 3
    F = cumtrapz(u1_bdy, s_eval, initial=0)
    fig_av, ax_av = plt.subplots()
    ax_av.plot(1 / s_eval[s_mask], s_eval[s_mask] ** 2 * u1_bdy[s_mask], 1 / s_eval[s_mask], 1 - F[s_mask])
    ax_av.axvline(1, color='k')
    ax_av.set_xlabel('$v^*$')
    ax_av.set_ylabel('Probability density')
    fig_av.savefig(plot_dir + filename + '.png')
    if show_plot:
        plt.show()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]

    main(filename, show_plot=True)
