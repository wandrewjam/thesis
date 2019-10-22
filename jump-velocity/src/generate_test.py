import numpy as np
from scipy.misc import factorial
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from simulate import experiment


def get_steps_dwells(y, t):
    trans = (y[2:] != y[1:-1]) + (y[:-2] != y[1:-1])
    trans = np.append(np.array([False]), trans)
    trans = np.append(trans, False)
    y_temp = y[trans]
    t_temp = t[trans]
    steps = y_temp[2::2] - y_temp[:-2:2]
    dwell = t_temp[1::2] - t_temp[:-1:2]
    return steps, dwell


def multiple_experiments(a, c, eps1, eps2, num_expt):
    b, d = 1 - a, 1 - c
    rate_a, rate_b = a / eps1, b / eps1
    rate_c, rate_d = c / eps2, d / eps2
    expts = [experiment(rate_a, rate_b, rate_c, rate_d, filter=False)
             for _ in range(num_expt)]

    ys = [expt[0] for expt in expts]
    ts = [expt[1] for expt in expts]
    vels = [1/(t[-1]) for t in ts]
    steps_dwells = [get_steps_dwells(y, t) for (y, t) in zip(ys, ts)]
    steps, dwells = np.zeros(shape=(0,)), np.zeros(shape=(0,))
    num_dwells = [len(dwell) for _, dwell in steps_dwells]
    for step, dwell in steps_dwells:
        steps = np.append(steps, step)
        dwells = np.append(dwells, dwell)

    return vels, steps, dwells, num_dwells


if __name__ == '__main__':
    a, c = .5, .2
    eps1, eps2 = .1, np.inf
    num_expt = 4096
    vels, steps, dwells, num_dwells = multiple_experiments(a, c, eps1, eps2,
                                                           num_expt)
    dwell_counts = np.bincount(num_dwells)
    plt.plot(np.arange(len(dwell_counts)),
             dwell_counts/np.sum(dwell_counts, dtype=float), 'o')
    pois_pdf = (np.exp(-(1 - a)/eps1 - (1 - c)/eps2)
                * ((1 - a)/eps1 + (1 - c)/eps2)**np.arange(len(dwell_counts))
                / factorial(np.arange(len(dwell_counts))))
    plt.plot(np.arange(len(dwell_counts)), pois_pdf, 'o')
    plt.show()
    print(dwell_counts)
    expect_dwell = pois_pdf[:13]*np.sum(dwell_counts)
    expect_dwell = np.append(expect_dwell, np.sum(dwell_counts)
                             - np.sum(expect_dwell))
    print(expect_dwell)
    obs_dwell = dwell_counts[:13]
    obs_dwell = np.append(obs_dwell, np.sum(dwell_counts[13:]))
    print(chisquare(obs_dwell, expect_dwell, ddof=1))
    # import os
    # os.chdir(os.path.expanduser('~/thesis/jump-velocity'))
    # filename = 'test4'
    # sim_dir = 'dat-files/simulations/'
    # np.savetxt(sim_dir + filename + '-step.dat', steps)
    # np.savetxt(sim_dir + filename + '-vel.dat', vels)
    # np.savetxt(sim_dir + filename + '-dwell.dat', dwells)
