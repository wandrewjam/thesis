import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, geom


def escape_experiment(rate_a, rate_b, rate_c, rate_d, escape):
    assert np.minimum(rate_a, rate_b) > 0
    assert escape > 0

    while True:
        t = np.zeros(shape=1)
        y = np.zeros(shape=1)
        state = np.zeros(shape=1, dtype='int')
        if rate_c == 0:
            assert rate_d == 0
            while y[-1] < 1:
                if state[-1] == 0:
                    total_rate = rate_b + escape
                    dt = np.random.exponential(scale=1. / total_rate)
                    r = np.random.rand(1)
                    if r < rate_b / total_rate:
                        t = np.append(t, t[-1] + dt)
                        y = np.append(y, y[-1] + dt)
                        state = np.append(state, 1)
                    else:
                        break
                elif state[-1] == 1:
                    dt = np.random.exponential(scale=1. / rate_a)
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
        if not filter:
            break
        if y.shape[0] > 1:  # Filter out plts that don't bind
            break

    return y, t, state


def simple_experiment(rate_b, escape, num_expts):
    dt1 = np.random.exponential(1. / rate_b, size=num_expts)
    dt2 = np.random.exponential(1. / escape, size=num_expts)
    return dt1[dt1 < dt2]


if __name__ == '__main__':
    num_expts = 2**16
    kon = 10.
    koff = 20.
    kes = 10.
    expts = [escape_experiment(koff, kon, 0, 0, kes) for _ in range(num_expts)]
    init_steps = list()
    steps = list()
    nsteps = list()
    mov_time = list()

    for ex in expts:
        t = ex[1]
        init_steps.append(t[1] - t[0])
        steps.extend(t[1::2] - t[:-1:2])
        mov_time.append(np.sum(t[1::2] - t[:-1:2]))
        nsteps.append((len(t) - 1) / 2)

    # Compare the distribution of initial steps. Expect exponential w/ rate kon
    xi_cdf = np.append([0], np.sort(init_steps))
    yi_cdf = np.linspace(0, 1, num=len(init_steps)+1)

    xj_cdf = np.append([0], np.sort(steps))
    yj_cdf = np.linspace(0, 1, num=len(steps)+1)

    xi_plot = np.linspace(0, xi_cdf[-1], num=512)
    yi_plot = expon.cdf(xi_plot, scale=1/(kon + kes))

    plt.plot(xi_cdf, yi_cdf)
    plt.plot(xj_cdf, yj_cdf)
    plt.plot(xi_plot, yi_plot)
    plt.title('Initial steps vs exponential distribution')
    plt.show()

    # Compare moving time with product of geometric and exponential rvs
    x_cdf = np.append([0], np.sort(mov_time))
    y_cdf = np.linspace(0, 1, num=len(mov_time)+1)

    n = geom.rvs(p=kes / (kes + kon), size=num_expts)
    m = [el * np.sum(expon.rvs(scale=1/(kon + kes), size=el)) for el in n]
    xplot = np.append([0], np.sort(m))
    yplot = np.linspace(0, 1, num=len(m) + 1)

    plt.plot(x_cdf, y_cdf)
    plt.plot(xplot, yplot)
    plt.title('Moving time vs product distribution')
    plt.show()

    # Compare no. of steps with geometric distribution
    a = kon / (kon + kes)
    x2plot = np.arange(1, np.amax(nsteps) + 1)
    plt.bar(x2plot, np.bincount(nsteps)[1:] / float(num_expts))
    plt.plot(x2plot, geom.pmf(x2plot, p=1 - a), 'r*')
    plt.title('No. of steps vs geometric distribution')
    plt.show()

    # dt = simple_experiment(2., 1., num_expts)
    # x_cdf = np.append([0], np.sort(dt))
    # y_cdf = np.linspace(0, 1, num=len(dt)+1)
    # xplot = np.linspace(0, x_cdf[-1], num=512)
    # yplot = expon.cdf(xplot, scale=1./3.)
    # plt.plot(x_cdf, y_cdf)
    # plt.plot(xplot, yplot)
    # plt.show()
