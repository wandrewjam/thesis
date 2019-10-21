import numpy as np
from simulate_modified import modified_experiment


def get_steps_dwells(y, t):
    trans = (y[2:] != y[1:-1]) + (y[:-2] != y[1:-1])
    trans = np.append(np.array([False]), trans)
    trans = np.append(trans, False)
    y_temp = y[trans]
    t_temp = t[trans]
    steps = y_temp[2::2] - y_temp[:-2:2]
    dwell = t_temp[1::2] - t_temp[:-1:2]
    return steps, dwell


def multiple_experiments(alpha, beta, gamma, delta, eta, lam, num_expt):
    expts = [modified_experiment(alpha, beta, gamma, delta, eta, lam)
             for _ in range(num_expt)]

    ys = [expt[0] for expt in expts]
    ts = [expt[1] for expt in expts]
    vels = [1/(t[-1]) for t in ts]
    steps_dwells = [get_steps_dwells(y, t) for (y, t) in zip(ys, ts)]
    steps, dwells = np.zeros(shape=(0,)), np.zeros(shape=(0,))
    for step, dwell in steps_dwells:
        steps = np.append(steps, step)
        dwells = np.append(dwells, dwell)

    return vels, steps, dwells


if __name__ == '__main__':
    alpha = 10
    beta = 10
    gamma = 10
    delta = 10
    eta = 10
    lam = 100
    num_expt = 1024
    vels, steps, dwells = multiple_experiments(alpha, beta, gamma, delta, eta,
                                               lam, num_expt)

    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))
    filename = 'testmod'
    sim_dir = 'dat-files/simulations/'
    np.savetxt(sim_dir + filename + '-step.dat', steps)
    np.savetxt(sim_dir + filename + '-vel.dat', vels)
    np.savetxt(sim_dir + filename + '-dwell.dat', dwells)
