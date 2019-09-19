import numpy as np


def main(scale):
    steps = np.random.exponential(scale=scale, size=64)

    sim_dir = 'dat-files/simulations/'
    np.savetxt(sim_dir + 'test-step.dat', np.array(steps))


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    scale = 1
    main(scale)
