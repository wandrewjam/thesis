import numpy as np
from scipy.stats import expon, gamma, truncnorm, uniform


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


def modified_experiment(alpha, beta, gam, delta, eta, lam, dwell_type):
    """Run a single experiment of the modified jump-velocity process

    This experiment excludes platelets that pass through the domain
    without binding.

    Parameters
    ----------
    alpha
    beta
    gam
    delta
    eta
    lam

    Returns
    -------

    """

    if dwell_type == 'exp':
        rv = expon(scale = 1./lam)
    elif dwell_type == 'unif':
        rv = uniform(scale=2./lam)
    elif dwell_type == 'gamma':
        rv = gamma(a=3, scale=1 / (3 * lam))
    elif dwell_type == 'norm':
        myclip_a, myclip_b = 0, 2./lam
        my_mean, my_std = 1./lam, 1./(2*lam)
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        rv = truncnorm(a, b, my_mean, my_std)
    else:
        raise ValueError('dwell_type is not valid')

    while True:
        t = np.zeros(shape=1)
        y = np.zeros(shape=1)
        state = np.zeros(shape=1, dtype='int')
        while y[-1] < 1:
            if state[-1] == 0:
                dt = np.random.exponential(1/beta)
                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1] + dt)
                state = np.append(state, 1)
            elif state[-1] == 1:
                dt = np.random.exponential([1/alpha, 1/delta])
                t = np.append(t, t[-1] + dt.min())
                y = np.append(y, y[-1])
                state = np.append(state, dt.argmin() * 2)  # Choose the correct
                                                           # state transition
            elif state[-1] == 2:
                dt = np.random.exponential([1 / gam, 1 / eta])
                t = np.append(t, t[-1] + dt.min())
                y = np.append(y, y[-1])
                state = np.append(state, (dt.argmin() * 2) + 1)
            elif state[-1] == 3:
                # Choose the time spent in the transition state
                # dt = np.random.exponential(1/lam)
                dt = rv.rvs()

                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1] + dt)
                state = np.append(state, 1)
            else:
                raise ValueError('State must be between 0 and 3')

        if y.shape[0] > 2:
            break

    excess = y[-1] - 1
    y[-1] -= excess
    t[-1] -= excess
    return y, t


def multiple_mod_experiments(alpha, beta, gamma, delta, eta, lam, num_expt, dwell_type):
    """Run many trials of the modified rolling experiment

    Parameters
    ----------
    alpha
    beta
    gamma
    delta
    eta
    lam
    num_expt

    Returns
    -------

    """
    expts = [modified_experiment(alpha, beta, gamma, delta, eta, lam, dwell_type)
             for _ in range(num_expt)]
    vels = [1/expt[1][-1] for expt in expts]
    return vels


def main(filename, alpha, beta, gamma, delta, eta, lam, num_expt, dwell_type='exp'):
    vels = multiple_mod_experiments(alpha, beta, gamma, delta, eta, lam,
                                    num_expt, dwell_type)

    sim_dir = 'dat-files/simulations/'
    np.savetxt(sim_dir + filename + '-{}-sim.dat'.format(dwell_type),
               np.sort(vels))
    return None


if __name__ == '__main__':
    import os
    import sys

    filename = sys.argv[1]
    dwell_type = sys.argv[2]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    pars = read_parameter_file(filename)
    main(dwell_type=dwell_type, **pars)
