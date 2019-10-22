import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, chi2
from generate_test import get_steps_dwells


def experiment_corrected(rate_a, rate_b, rate_c, rate_d, length):
    """Run a single long experiment of the modified j-v process

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
    length : float or int

    Returns
    -------

    """

    assert np.minimum(rate_a, rate_b) > 0
    t = np.zeros(shape=1)
    y = np.zeros(shape=1)
    state = np.zeros(shape=1, dtype='int')
    if rate_c == 0:
        assert rate_d == 0
        while y[-1] < length:
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
        while y[-1] < length:
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

    excess = y[-1] - 1
    y[-1] -= excess
    t[-1] -= excess
    return y, t


def main():
    a, eps1 = .5, .1
    c, eps2 = .3, 1.
    length = 2**14.

    b, d = 1 - a, 1 - c
    rate_a, rate_b = a / eps1, b / eps1
    rate_c, rate_d = c / eps2, d / eps2
    step_rate = rate_b + rate_d
    y, t = experiment_corrected(rate_a, rate_b, rate_c, rate_d, length)
    steps, dwell = get_steps_dwells(y, t)

    xplot = np.linspace(start=0, stop=steps.max(), num=512)
    plt.hist(steps, density=True)
    plt.plot(xplot, step_rate*np.exp(-step_rate*xplot))
    plt.show()

    s = np.sort(steps)
    s = np.insert(s, values=0, obj=0)
    y_cdf = np.arange(0., len(s))/(len(s) - 1)
    plt.step(s, y_cdf)
    plt.plot(xplot, 1 - np.exp(-step_rate*xplot))
    plt.show()

    cdf_fun = lambda x: 1 - np.exp(-step_rate * x)
    print(kstest(steps, cdf_fun))
    n = len(steps)
    print((2*n*np.mean(steps)/chi2.ppf(0.975, df=2*n),
           2*n*np.mean(steps)/chi2.ppf(0.025, df=2*n)))
    print(1./step_rate)


if __name__ == '__main__':
    main()
