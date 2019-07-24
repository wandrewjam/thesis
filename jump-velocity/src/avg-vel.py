import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from jv import read_parameter_file


def main(filename, show_plot=False):
    npz_dir = 'npz-files/'
    plot_dir = 'plots/'
    data = np.load(npz_dir + filename + '.npz')
    y, s_eval = data['y'], data['s_eval']
    u0_bdy, u1_bdy = data['u0_bdy'], data['u1_bdy']
    pars = read_parameter_file(filename)

    s_mask = s_eval > 4./5
    F = cumtrapz(u0_bdy + u1_bdy, s_eval, initial=0)
    fig_av, ax_av = plt.subplots()
    ax_av.plot(1 / s_eval[s_mask], s_eval[s_mask] ** 2 * u1_bdy[s_mask], label='Velocity PDF of filtered plts')
    ax_av.plot(1 / s_eval[s_mask], 1 - F[s_mask], label='Velocity CDF of all plts')
    ax_av.axvline(1, color='k')
    ax_av.legend(loc='best')
    ax_av.set_xlabel('$v^*$')
    ax_av.set_ylabel('Probability density')
    ax_av.set_title('$\\epsilon_1 = {}$, $\\epsilon_2 = {}$, $a = {}$, $c = {}$'.format(pars['eps1'], pars['eps2'], pars['a'], pars['c']))
    fig_av.savefig(plot_dir + filename + '.png')
    if show_plot:
        plt.show()

    dat_dir = 'dat-files/distributions/'
    v_dat = np.linspace(1/s_eval[-1], 1.5, num=1001)
    f_dat = np.interp(v_dat, (1 / s_eval[s_mask])[::-1],
                      (s_eval[s_mask] ** 2 * u1_bdy[s_mask])[::-1])
    F_dat = np.interp(v_dat, (1 / s_eval[s_mask])[::-1], (1 - F[s_mask])[::-1])
    np.savetxt(dat_dir + filename + '-dst.dat',
               np.stack((v_dat, f_dat, F_dat), axis=-1))


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    main(filename)
