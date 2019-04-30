import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from parameter_sweep import solve_along_chars

### I was working on correcting the axes of the right columns, and labeling the curves

if __name__ == '__main__':
    z0 = np.linspace(-5, 5, num=501)
    omegas = np.linspace(0, 200, num=501)[1:]
    kappas = np.array([5e-1, 1., 2.])
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
                           figsize=(6, 6))
    omfs = [omegas + result[1]/xi for result in results]
    m = [np.amax(omf) for omf in omfs]
    m = np.amax(m)

    for (j, result) in enumerate(results):
        ax[j, 0].plot(np.hstack([0, omegas]), np.hstack([0, result[1]]))
        ax[j, 0].set_ylabel('Torque $\\tau$')
        ax[j, 1].plot(np.hstack([0, omegas + result[1]/xi]), np.hstack([0, omegas]))
        ax[j, 1].plot([0, m], [0, m], 'k--')
        ax[j, 1].set_ylabel('Rotation rate $\\omega$')
    ax[-1, 0].set_xlabel('Rotation rate $\\omega$')
    ax[-1, 1].set_xlabel('Applied rotation $\\omega_f$')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=kappas.shape[0], sharex='col',
                           figsize=(4, 6))
    xis = [1e-5, 2e-5, 5e-5]
    omfs = [omegas + results[2][1]/xi for xi in xis]
    m = [np.amax(omf) for omf in omfs]
    m = np.amax(m)

    for (j, omf) in enumerate(omfs):
        ax[j].plot(np.hstack([0, omf]), np.hstack([0, omegas]))
        ax[j].plot([0, m], [0, m], 'k--')
        ax[j].set_ylabel('Rotation rate $\\omega$')
    ax[-1].set_xlabel('Applied rotation $\\omega_f$')
    plt.tight_layout()
    plt.show()
