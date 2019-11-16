import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from generate_test import get_steps_dwells


def modified_experiment_corrected(alpha, beta, gamma, delta, eta, lam, length):
    """Run a single long experiment of the modified j-v process

    Parameters
    ----------
    alpha
    beta
    gamma
    delta
    eta
    lam

    Returns
    -------

    """
    t = np.zeros(shape=1)
    y = np.zeros(shape=1)
    state = np.zeros(shape=1, dtype='int')
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
            dt = np.random.exponential([1/gamma, 1/eta])
            t = np.append(t, t[-1] + dt.min())
            y = np.append(y, y[-1])
            # Choose the correct state transition
            state = np.append(state, (dt.argmin() * 2) + 1)
        elif state[-1] == 3:
            dt = np.random.exponential(1/lam)
            t = np.append(t, t[-1] + dt)
            y = np.append(y, y[-1] + dt)
            state = np.append(state, 1)
        else:
            raise ValueError('State must be between 0 and 3')

    return y, t


def main():
    alpha, beta = 10., 10.
    gamma, delta = 10., 10.
    eta, lam = 10., 10.
    length = 2**12.

    y, t = modified_experiment_corrected(alpha, beta, gamma, delta, eta,
                                         lam, length)
    steps, dwell = get_steps_dwells(y, t)
    pdf_fun = lambda x: ((alpha * beta * (gamma + eta) * np.exp(-beta * x)
                          + delta**2 * eta * np.exp(-delta * x))
                         / (alpha * (gamma + eta) + delta * eta))

    cdf_fun = lambda x: 1 - ((alpha * (gamma + eta) * np.exp(-beta * x)
                              + delta * eta * np.exp(-delta * x))
                             / (alpha * (gamma + eta) + delta * eta))

    xplot = np.linspace(start=0, stop=steps.max(), num=512)
    plt.hist(steps, density=True)
    plt.plot(xplot, pdf_fun(xplot))
    plt.show()

    s = np.sort(steps)
    s = np.insert(s, values=0, obj=0)
    y_cdf = np.arange(0., len(s))/(len(s) - 1)
    plt.step(s, y_cdf)
    plt.plot(xplot, cdf_fun(xplot))
    plt.show()

    print(kstest(steps, cdf_fun))


if __name__ == '__main__':
    main()