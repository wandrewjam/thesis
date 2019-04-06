import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == '__main__':
    filename = sys.argv[1]
    data = np.load(filename)

    M = data['M']
    T = data['T']
    kap_vec = data['kap_vec']
    omegas = data['omegas']

    low, high = 1e-3, 1e-2
    filtered = (M > low) * (M < high)

    plt.pcolormesh(omegas, kap_vec[::2], filtered[:, :-1])
    plt.xlabel('$\\omega$')
    plt.ylabel('$\\kappa$')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
