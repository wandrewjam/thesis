import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from parameter_sweep import solve_along_chars


if __name__ == '__main__':
    z0 = np.linspace(-5, 5, num=1001)
    omegas = np.linspace(0, 100, num=201)[1:]
    kappas = np.array([1e-1, 5e-1, 1., 2.])
    eta = 1e4
    delta = 16
    xi = 1e-5

    pool = mp.Pool(processes=3)
    results = [pool.apply_async(solve_along_chars, args=(z0, omegas, kappa,
                                                         eta, delta))
               for kappa in kappas]
    results = [res.get() for res in results]

    # results = list()
    # for kappa in kappas:
    #     results.append(solve_along_chars(z0, omegas, kappa, eta, delta))

    fig, ax = plt.subplots(nrows=kappas.shape[0], ncols=2, sharex='col',
                           figsize=(8, 9))
    for (j, result) in enumerate(results):
        ax[j, 0].plot(omegas, result[1])
        ax[j, 0].set_ylabel('Torque $\\tau$')
        m = np.amax(omegas + result[1]/xi)
        ax[j, 1].plot(omegas + result[1]/xi, omegas)
        ax[j, 1].plot([0, m], [0, m], 'k--')
        ax[j, 1].set_ylabel('Rotation rate $\\omega$')
    ax[-1, 0].set_xlabel('Rotation rate $\\omega$')
    ax[-1, 1].set_xlabel('Applied rotation $\\omega_f$')
    plt.tight_layout()
    plt.show()
