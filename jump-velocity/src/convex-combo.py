import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz


def main(filename1, filename2, gamma):
    npz_dir = 'npz-files/'
    plot_dir = 'plots/'
    data0 = np.load(npz_dir + filename1 + '.npz')
    data1 = np.load(npz_dir + filename2 + '.npz')

    s_eval = data0['s_eval']
    u0_bdy0, u1_bdy0 = data0['u0_bdy'], data0['u1_bdy']
    u0_bdy1, u1_bdy1 = data1['u0_bdy'], data1['u1_bdy']

    s_mask = s_eval > 4./5
    F = cumtrapz(gamma*(u0_bdy0 + u1_bdy0) + (1 - gamma)*(u0_bdy1 + u1_bdy1), s_eval, initial=0)
    fig_av, ax_av = plt.subplots()
    ax_av.plot(1 / s_eval[s_mask], s_eval[s_mask] ** 2 * (gamma*u1_bdy0[s_mask] + (1-gamma)*u1_bdy1[s_mask]), label='Velocity PDF of filtered plts')
    ax_av.plot(1 / s_eval[s_mask], 1 - F[s_mask], label='Velocity CDF of all plts')
    ax_av.axvline(1, color='k')
    ax_av.legend(loc='upper right')
    ax_av.set_xlabel('$v^*$')
    ax_av.set_ylabel('Probability density')
    fig_av.savefig(plot_dir + filename1 + '_' + filename2 + '_' + str(gamma) + '.png')
    plt.show()


if __name__ == '__main__':
    import os
    import sys
    filename1, filename2 = sys.argv[1:3]
    gamma = float(sys.argv[3])
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    main(filename1, filename2, gamma)
