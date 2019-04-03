import matplotlib.pyplot as plt
import numpy as np
import sys
from itertools import combinations


if __name__ == '__main__':
    filepath = '../data/toy-model/'
    filename = sys.argv[1]
    data = np.load(filepath + filename)

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
    for (ax1, ax2) in combinations(range(3), 2):
        slices.append(np.average(np.average(filtered, axis=ax2,
                                            weights=weights[ax2]),
                                 axis=ax1, weights=weights[ax1]))

    for (j, arr) in zip(reversed(range(3)), slices):
        plt.pcolormesh(omegas, pars[j], arr[:, :-1])
        plt.xlabel('$\\omega$')
        plt.ylabel(par_names[j])
        plt.yscale('log')
        plt.colorbar()
        plt.show()
