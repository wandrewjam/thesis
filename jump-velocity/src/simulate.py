import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from src.jv import solve_pde, delta_h
from src.bootstrap import bootstrap
from src.mle import fit_models, change_vars


# def read_parameter_file(filename):
#     txt_dir = 'par-files/'
#     parlist = [('filename', filename)]
#
#     with open(txt_dir + filename + '.txt') as f:
#         while True:
#             command = f.readline().split()
#             if len(command) < 1:
#                 continue
#             if command[0] == 'done':
#                 break
#
#             key, value = command
#             if key == 'trials':
#                 parlist.append((key, int(value)))
#             else:
#                 parlist.append((key, float(value)))
#     return dict(parlist)


def experiment(rate_a, rate_b, rate_c, rate_d):
    """Runs a single jump-velocity experiment based on the given rates

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
    excess = y[-1] - 1
    y[-1] -= excess
    t[-1] -= excess
    return 1 / t[-1]


def plot_experiments(vels, reduced=None, full_model=None):
    """Plots histogram and ECDF of average velocity data

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    reduced : callable
    full_model : callable

    Returns
    -------
    None
    """
    if reduced is not None or full_model is not None:
        x_plot = np.linspace(0, 1.5, num=500)[1:]

    density_fig, density_ax = plt.subplots()
    density_ax.hist(vels, density=True, label='Sample data')

    if reduced is not None:
        density_ax.plot(x_plot, reduced(x_plot), label='Adiabatic reduction')
    if full_model is not None:
        density_ax.plot(x_plot, full_model(x_plot), label='Full model')
    density_ax.legend(loc='best')
    plt.show()

    x_cdf = np.sort(vels)
    x_cdf = np.insert(x_cdf, 0, 0)
    y_cdf = np.linspace(0, 1, len(x_cdf))
    x_cdf = np.append(x_cdf, 1.5)
    y_cdf = np.append(y_cdf, 1)

    cdf_fig, cdf_ax = plt.subplots()
    cdf_ax.step(x_cdf, y_cdf, where='post', label='Sample data')
    cdf_ax.plot(x_plot[::-1],
                1 + cumtrapz(reduced(x_plot[::-1]), x_plot[::-1], initial=0),
                label='Adiabatic reduction')
    cdf_ax.plot(x_plot[::-1],
                1 + cumtrapz(full_model(x_plot[::-1]), x_plot[::-1],
                             initial=0),
                label='Full model')
    cdf_ax.legend(loc='best')
    plt.show()


def main(a=.5, c=.2, eps1=.1, eps2=1, num_expt=1000):
    b, d = 1 - a, 1 - c

    rate_a, rate_b = a / eps1, b / eps1
    rate_c, rate_d = c / eps2, d / eps2

    vels = list()
    for i in range(num_expt):
        vels.append(experiment(rate_a, rate_b, rate_c, rate_d))

    def reduced(v, a_reduced, eps_reduced):
        b_reduced = 1 - a_reduced
        return (1 / np.sqrt(4 * np.pi * eps_reduced
                            * a_reduced * b_reduced * v ** 3)
                * (a_reduced + (v - a_reduced) / 2)
                * np.exp(-(v - a_reduced) ** 2 / (4 * eps_reduced * a_reduced
                                                  * b_reduced * v)))

    N = 1000
    s_max = 50
    s_eval = np.linspace(0, s_max, num=s_max * N + 1)
    h = 1. / N
    scheme = None
    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N))

    reduced_fit, full_fit = fit_models(vels)
    a_rfit, eps_rfit = change_vars(reduced_fit, forward=False)[:, 0]

    a_ffit, eps_ffit = change_vars(full_fit, forward=False)[:, 0]
    sol_fit = solve_pde(s_eval, p0, h, eps_ffit, np.inf, a_ffit, 1 - a_ffit,
                        1, 0, scheme)[3]
    full_model = interp1d(1 / s_eval[1:], sol_fit[1:] * s_eval[1:] ** 2,
                          bounds_error=False, fill_value=0)
    plot_experiments(vels, reduced=lambda v: reduced(v, a_rfit, eps_rfit),
                     full_model=full_model)

    parameter_trials = bootstrap(vels, boot_trials=4, proc=2)

    print(np.array([[a_rfit, a_ffit], [eps_rfit, eps_ffit]]))
    print(a, eps1)
    print(parameter_trials)
    print(parameter_trials.shape)
    print(np.percentile(parameter_trials, q=(2.5, 97.5), axis=-1))


if __name__ == '__main__':
    # import sys
    # filename = sys.argv[1]
    #
    # pars = read_parameter_file(filename)
    # main(**pars)
    main(eps1=.1, eps2=np.inf, num_expt=1000)
