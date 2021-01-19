import numpy as np
import matplotlib.pyplot as plt
from sphere_integration_utils import wall_stokeslet_integrand


if __name__ == '__main__':
    x_arr = np.logspace(-1, -3, num=100)
    force = np.array([-1, 0, 0])
    eps_arr = np.array([.1, .05, .01])

    f0 = 1.75e9
    tau = 2000.
    rep_force = f0 * (tau * np.exp(-tau * x_arr)) / (1 - np.exp(-tau * x_arr))

    ef = []
    for eps in eps_arr:
        f = []
        for x in x_arr:
            center = np.array([x, 0, 0])
            f.append(-1. / wall_stokeslet_integrand(center, center, eps,
                                                    force)[0])
        ef.append(f)

    ef = np.array(ef)
    plt.semilogy(x_arr, ef.T, x_arr, rep_force)
    plt.xlabel('Distance from the wall')
    plt.ylabel('Force')
    plt.ylim([10**-7, 10**4])
    plt.legend(['$\\epsilon = 0.1$', '$\\epsilon = 0.05$',
                '$\\epsilon = 0.01$'])
    plt.show()
