import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
from itertools import combinations


if __name__ == '__main__':
    filename = sys.argv[1]
    data = np.load(filename)

    M = data['M']
    T = data['T']
    kap_vec = data['kap_vec']
    eta_vec = data['eta_vec']
    del_vec = data['del_vec']
    omegas = data['omegas']

    low, high = 1e-4, 1e-2
    filtered = (M > low) * (M < high)

    pars = [kap_vec[::2], eta_vec[::2], del_vec[::2]]
    par_names = ['$\\kappa$', '$\\eta$', '$\\delta$']
    weights = [arr[1:] - arr[:-1] for arr in pars]

    slices = list()
    for axs1 in combinations(range(3), 2):
        slices.append(np.average(filtered, axis=axs1))

    fig0, axs0 = plt.subplots(ncols=1, nrows=3, sharex='all', figsize=(5, 9))

    for (j, arr) in zip(reversed(range(3)), slices):
        qm = axs0[j].pcolormesh(omegas, pars[j], arr[:, :-1], vmin=0, vmax=1)
        axs0[j].set_ylabel(par_names[j])
        axs0[j].set_yscale('log')
        fig0.colorbar(qm, ax=axs0[j])
    axs0[-1].set_xlabel('$\\omega$')
    fig0.tight_layout()
    plt.show()

    fig1, axs1 = plt.subplots(ncols=2, nrows=2, sharey='all', sharex='all',
                              figsize=(8, 8))

    levels = [1e-4, 1e-3, 1e-2, 1]

    cs = axs1[0, 0].contourf(omegas, kap_vec[1::2], M[:, 0, 0, :], levels,
                             norm=LogNorm())
    sp_title = '$\\eta = {:g}$, $\\delta = {:g}$'
    axs1[0, 0].set_title(sp_title.format(eta_vec[0], del_vec[0]))
    axs1[1, 0].contourf(omegas, kap_vec[1::2], M[:, -1, 0, :], levels,
                        norm=LogNorm())
    axs1[1, 0].set_title(sp_title.format(eta_vec[-1], del_vec[0]))
    axs1[0, 1].contourf(omegas, kap_vec[1::2], M[:, 0, -1, :], levels,
                        norm=LogNorm())
    axs1[0, 1].set_title(sp_title.format(eta_vec[0], del_vec[-1]))
    axs1[1, 1].contourf(omegas, kap_vec[1::2], M[:, -1, -1, :], levels,
                        norm=LogNorm())
    axs1[1, 1].set_title(sp_title.format(eta_vec[-1], del_vec[-1]))

    axs1[0, 0].set_ylabel('$\\kappa$')
    axs1[1, 0].set_ylabel('$\\kappa$')
    axs1[1, 0].set_xlabel('$\\omega$')
    axs1[1, 1].set_xlabel('$\\omega$')

    axs1[0, 0].set_yscale('log')

    fig1.subplots_adjust(right=0.8)
    cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
    fig1.colorbar(cs, cax=cbar_ax)
    plt.show()

    print(np.unique(filtered, return_counts=True))
    print(np.unique(np.nonzero(filtered)[0], return_counts=True), kap_vec[::2][0], kap_vec[::2][24])
    print(np.unique(np.nonzero(filtered)[1], return_counts=True))
    print(np.unique(np.nonzero(filtered)[2], return_counts=True))
    print(filtered.size)
