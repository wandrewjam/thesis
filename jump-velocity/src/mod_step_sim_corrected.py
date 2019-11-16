import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import expon, gamma, truncnorm, uniform
from generate_test import get_steps_dwells
from simulate_modified import read_parameter_file


def modified_experiment_corrected(alpha, beta, gam, delta, eta, lam, length,
                                  dwell_type):
    """Run a single long experiment of the modified j-v process

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
    t = np.zeros(shape=1)
    y = np.zeros(shape=1)
    state = np.zeros(shape=1, dtype='int')

    if dwell_type == 'exp':
        rv = expon(scale = 1./lam)
    elif dwell_type == 'unif':
        rv = uniform(scale=2./lam)
    elif dwell_type == 'gamma':
        rv = gamma(a=3, scale=1 / (3 * lam))
    elif dwell_type == 'norm':
        myclip_a, myclip_b = 0, np.inf
        my_mean, my_std = 1./lam, 1./(2*lam)
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        rv = truncnorm(a, b, my_mean, my_std)
    else:
        if dwell_type != 'delta':
            raise ValueError('dwell_type is not valid')

    while y[-1] < length:
        if state[-1] == 0:
            dt = np.random.exponential(1/beta)
            t = np.append(t, t[-1] + dt)
            y = np.append(y, y[-1] + dt)
            state = np.append(state, 1)
        elif state[-1] == 1:
            dt = np.random.exponential([1/alpha, 1/delta])
            t = np.append(t, t[-1] + dt.min())
            y = np.append(y, y[-1])
            # Choose the correct state transition
            state = np.append(state, dt.argmin() * 2)
        elif state[-1] == 2:
            dt = np.random.exponential([1 / gam, 1 / eta])
            t = np.append(t, t[-1] + dt.min())
            y = np.append(y, y[-1])
            # Choose the correct state transition
            state = np.append(state, (dt.argmin() * 2) + 1)
        elif state[-1] == 3:
            if dwell_type == 'delta':
                dt = 1./lam
            else:
                dt = rv.rvs()
            t = np.append(t, t[-1] + dt)
            y = np.append(y, y[-1] + dt)
            state = np.append(state, 1)
        else:
            raise ValueError('State must be between 0 and 3')

    return y, t


def main(filename, alpha, beta, gamma, delta, eta, lam, num_expt,
         dwell_type='exp', plot=False):
    length = 2**10.

    y, t = modified_experiment_corrected(alpha, beta, gamma, delta, eta,
                                         lam, length, dwell_type)
    steps, dwell = get_steps_dwells(y, t)

    pdf_fun = lambda x: ((alpha * beta * (gamma + eta) * np.exp(-beta * x)
                          + delta * eta * lam * np.exp(-lam * x))
                         / (alpha * (gamma + eta) + delta * eta))

    cdf_fun = lambda x: 1 - ((alpha * (gamma + eta) * np.exp(-beta * x)
                              + delta * eta * np.exp(-lam * x))
                             / (alpha * (gamma + eta) + delta * eta))

    xplot = np.linspace(start=0, stop=steps.max(), num=512)
    s = np.sort(steps)
    s = np.insert(s, values=0, obj=0)
    y_cdf = np.arange(0., len(s))/(len(s) - 1)

    print(kstest(steps, cdf_fun))

    if plot:
        plt.hist(steps, density=True)
        plt.plot(xplot, pdf_fun(xplot))
        plt.show()

        plt.step(s, y_cdf)
        plt.plot(xplot, cdf_fun(xplot))
        plt.show()

    sim_dir = 'dat-files/simulations/'
    np.savetxt(sim_dir + filename + '-step.dat', np.sort(steps))
    np.savetxt(sim_dir + filename + '-{}-step.dat'.format(dwell_type),
               np.sort(steps[steps < 2./lam]))
    np.savetxt(sim_dir + filename + '-step-cdf.dat', np.array([s[::10],
                                                               y_cdf[::10]]).T)

    dst_dir = 'dat-files/distributions/'
    np.savetxt(dst_dir + filename + '-step-dst.dat',
               np.array([xplot, pdf_fun(xplot), cdf_fun(xplot)]).T)
    return None


if __name__ == '__main__':
    import os
    import sys

    filename = sys.argv[1]
    dwell_type = sys.argv[2]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    pars = read_parameter_file(filename)
    main(dwell_type=dwell_type, **pars)
