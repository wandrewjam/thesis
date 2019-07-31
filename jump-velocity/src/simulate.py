import numpy as np


def read_parameter_file(filename):
    txt_dir = 'par-files/experiments/'
    parlist = [('filename', filename)]

    with open(txt_dir + filename + '.txt') as f:
        while True:
            command = f.readline().split()
            if len(command) < 1:
                continue
            if command[0] == 'done':
                break

            key, value = command
            if key == 'num_expt':
                parlist.append((key, int(value)))
            else:
                parlist.append((key, float(value)))
    return dict(parlist)


def experiment(rate_a, rate_b, rate_c, rate_d):
    """Runs a single jump-velocity experiment based on the given rates

    Note that this experiment does not exclude platelets that pass
    through the domain without binding.

    Parameters
    ----------
    rate_a : float
        Value of rate a
    rate_b : float
        Value of rate b
    rate_c : float
        Value of rate c
    rate_d : float
        Value of rate d

    Returns
    -------
    avg_velocity : float
        Average velocity of the platelet across the domain
    """

    assert np.minimum(rate_a, rate_b) > 0
    while True:
        t = np.zeros(shape=1)
        y = np.zeros(shape=1)
        state = np.zeros(shape=1, dtype='int')
        if rate_c == 0:
            assert rate_d == 0
            while y[-1] < 1:
                if state[-1] == 0:
                    dt = np.random.exponential(scale=1 / rate_b)
                    t = np.append(t, t[-1] + dt)
                    y = np.append(y, y[-1] + dt)
                    state = np.append(state, 1)
                elif state[-1] == 1:
                    dt = np.random.exponential(scale=1 / rate_a)
                    t = np.append(t, t[-1] + dt)
                    y = np.append(y, y[-1])
                    state = np.append(state, 0)
                else:
                    raise ValueError('State must be 0 or 1')
        else:
            assert rate_d > 0
            while y[-1] < 1:
                if state[-1] == 0:
                    dt = np.random.exponential(scale=1 / (rate_b + rate_d))
                    t = np.append(t, t[-1] + dt)
                    y = np.append(y, y[-1] + dt)
                    r = np.random.rand(1)
                    if r < rate_b / (rate_b + rate_d):
                        state = np.append(state, 1)
                    else:
                        state = np.append(state, 2)
                elif state[-1] == 1:
                    dt = np.random.exponential(scale=1 / (rate_a + rate_d))
                    t = np.append(t, t[-1] + dt)
                    y = np.append(y, y[-1])
                    r = np.random.rand(1)
                    if r < rate_a / (rate_a + rate_d):
                        state = np.append(state, 0)
                    else:
                        state = np.append(state, 3)
                elif state[-1] == 2:
                    dt = np.random.exponential(scale=1 / (rate_b + rate_c))
                    t = np.append(t, t[-1] + dt)
                    y = np.append(y, y[-1])
                    r = np.random.rand(1)
                    if r < rate_c / (rate_b + rate_c):
                        state = np.append(state, 0)
                    else:
                        state = np.append(state, 3)
                elif state[-1] == 3:
                    dt = np.random.exponential(scale=1 / (rate_a + rate_c))
                    t = np.append(t, t[-1] + dt)
                    y = np.append(y, y[-1])
                    r = np.random.rand(1)
                    if r < rate_a / (rate_a + rate_c):
                        state = np.append(state, 2)
                    else:
                        state = np.append(state, 1)
                else:
                    raise ValueError('State must be one of 0, 1, 2, or 3')
        if y.shape[0] > 2:  # Filter out plts that don't bind
            break

    excess = y[-1] - 1
    y[-1] -= excess
    t[-1] -= excess
    return 1 / t[-1]


def main(a, c, eps1, eps2, num_expt, filename):
    b, d = 1 - a, 1 - c

    rate_a, rate_b = a / eps1, b / eps1
    rate_c, rate_d = c / eps2, d / eps2

    vels = list()
    for i in range(num_expt):
        vels.append(experiment(rate_a, rate_b, rate_c, rate_d))

    sim_dir = 'dat-files/simulations/'
    if eps2 == np.inf:
        header = 'two par'
    else:
        header = 'four par'
    np.savetxt(sim_dir + filename + '-sim.dat', np.sort(vels), header=header)


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    pars = read_parameter_file(filename)
    main(**pars)
