import numpy as np
import matplotlib.pyplot as plt


def bootstrap_experiment(steps, dwells, avg_vels, V, L=100.):
    np.random.seed()
    t = np.zeros(shape=1)
    y = np.zeros(shape=1)
    state = np.zeros(shape=1, dtype='int')
    while y[-1] < L:
        if state[-1] == 0:
            dy = np.random.choice(steps)
            dt = dy / V
            state = np.append(state, 1)
        elif state[-1] == 1:
            dt = np.random.choice(dwells)
            dy = 0
            state = np.append(state, 0)
        else:
            raise ValueError('State is invalid')
        t = np.append(t, t[-1] + dt)
        y = np.append(y, y[-1] + dy)

    excess = y[-1] - L
    y[-1] -= excess
    t[-1] -= excess / V
    return y, t


def main(filename):
    sim_dir = 'dat-files/simulations/'
    steps = np.loadtxt(sim_dir + filename + '-step.dat')
    dwells = np.loadtxt(sim_dir + filename + '-dwell.dat')
    avg_vels = np.loadtxt(sim_dir + filename + '-vel.dat')

    # v_max = np.amax(avg_vels)
    v_max = 10.
    experiments = [
        bootstrap_experiment(steps, dwells, avg_vels, v_max)
        for _ in range(1024)
    ]

    expt_vels = np.array([y[-1] / t[-1] for y, t in experiments])

    vel_sims = np.loadtxt(sim_dir + filename + 'samp-gamma-sim.dat')

    opacity = 0.8
    fig, ax = plt.subplots(ncols=2, figsize=[10., 4.], sharex='all')
    ax[0].hist(avg_vels, density=True, alpha=opacity)
    ax[0].hist(expt_vels, density=True, alpha=opacity)
    ax[0].hist(vel_sims * v_max, density=True, alpha=opacity)
    ax[0].legend(['Observed velocities', 'Bootstrapped velocities',
                  'Simulated velocities'])
    ax[0].set_xlim(0, 10)
    ax[0].set_xlabel('Average velocity $(\\mu m / s)$')
    ax[0].set_ylabel('Probability density')

    ax[1].step(np.append(0, np.sort(avg_vels)),
               np.append(0, (np.arange(0, avg_vels.size) + 1.)/avg_vels.size),
               where='post')
    ax[1].step(np.append(0, np.sort(expt_vels)),
               np.append(0, (np.arange(0, expt_vels.size) + 1.)/expt_vels.size),
               where='post')
    ax[1].step(np.append(0, np.sort(vel_sims * v_max)),
               np.append(0, (np.arange(0, (vel_sims * v_max).size)
                             + 1.) / (vel_sims * v_max).size),
               where='post')
    ax[1].legend(['Observed velocities', 'Bootstrapped velocities',
                  'Simulated velocities'])
    ax[1].set_xlabel('Average velocity $(\\mu m / s)$')
    ax[1].set_ylabel('Cumulative probability')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    filename = raw_input('Enter the filename: ')

    main(filename)
