import numpy as np
from jv import solve_pde, delta_h
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


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


# def main(trials, eps1, eps2, a, c, filename):
#     assert np.minimum(eps1, eps2) < np.inf
#     if np.maximum(eps1, eps2) == np.inf:
#         # Two-state problem
#         if eps1 == np.inf:
#             # Define a, b to be the nonzero reaction rates
#             eps1, a = eps2, c
#         steps = np.random.exponential(scale=eps1/a, size=int(a*trials/eps1))
#         b = 1 - a
#         pauses = np.random.exponential(scale=eps1/b, size=int(a*trials/eps1)-1)
#         step_times = np.cumsum(steps)
#         while step_times[-1] < trials:
#             step_times = np.append(step_times, step_times[-1]
#                                    + np.random.exponential(eps1/a))
#             pauses = np.append(pauses, np.random.exponential(eps1/b))
#         indices = np.searchsorted(step_times, np.arange(stop=trials)+1)
#         step_split = np.split(steps, indices)
#         pause_split = np.split(pauses, indices-1)
#         for step in step_split:
#
#     b, d = 1 - a, 1 - c
#     ka, kb = a / (1 + eps1 / eps2), b / (1 + eps1 / eps2)
#     kc, kd = c / (1 + eps2 / eps1), d / (1 + eps2 / eps1)
#     coeff = 1/eps1 + 1/eps2
#
#     y, j = np.zeros(shape=1), np.zeros(shape=1, dtype='int')
#     t = np.zeros(shape=1)
#
#     while True:
#         if j[-1] == 0:
#             r_sum = np.cumsum(coeff * np.array([kb, kd]))
#         dt = np.random.exponential(r_sum[-1])


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


def change_vars(p, forward=True, vectorized=False):
    """Convert between model parameters and fitting parameters

    Parameters
    ----------
    p : array_like
        Parameter values to convert
    forward : bool, optional
        If True, converts model parameters to fitting parameters. If
        False, converts fitting parameters to model parameters.
    vectorized : bool, optional
        Use a vectorized implementation of the function

    Returns
    -------
    ndarray
        Corresponding fitting (if forward is True) or model (if forward
        is False) parameters
    """
    if p.ndim < 2:
        p = p[:, None]

    result = np.zeros(shape=p.shape)
    if forward:
        result[0] = (2 * p[0] - 1) / (2 * p[0] * (1 - p[0]))
        result[1] = np.log(p[1])
    else:
        result[p != 0] = ((p[p != 0] - 1 + np.sqrt(p[p != 0] ** 2 + 1))
                          / (2 * p[p != 0]))
        result[p == 0] = 0.5
        result[1] = np.exp(p[1])
    return result


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


def fit_models(vels, initial_guess=None):
    """Fits the adiabatic reduction and full PDE model to data

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    initial_guess : array_like or None, optional
        Initial guess for the minimization procedure. If None, then
        infer a reasonable starting guess from the mean and standard
        deviation of the data

    Returns
    -------
    reduced_fit : ndarray
        Maximum likelihood parameters of the reduced model
    full_fit : ndarray
        Maximum likelihood parameters of the full PDE model
    """
    from scipy.optimize import minimize

    def q(y, s, a, eps):
        return (1 / np.sqrt(4 * np.pi * eps * a * (1 - a) * s)
                * (a + (y - a * s) / (2 * s))
                * np.exp(-(y - a * s) ** 2 / (4 * eps * a * (1 - a) * s)))

    def reduced_objective(p, v, vectorized=False):
        if vectorized:
            v = np.array(vels)[:, None, None]
            s = p[0][None, :, None]
            ap = np.select([s == 0, s != 0],
                           [0.5, (s - 1 + np.sqrt(s ** 2 + 1)) / (2 * s)])
            epsp = np.exp(p[1][None, None, :])
            return -np.sum(np.log(q(1, 1. / v, ap, epsp)), axis=0)
        else:
            v = np.array(vels)
            s = p[0]
            if s == 0:
                ap = .5
            else:
                ap = (s - 1 + np.sqrt(s ** 2 + 1)) / (2 * s)
            epsp = np.exp(p[1])
            return -np.sum(np.log(q(1, 1. / v, ap, epsp)))

    def full_objective(p, v, vmin):
        """ Log likelihood function of the full model

        Parameters
        ----------
        p
        v
        vmin

        Returns
        -------
        float
        """
        # Transform the parameters back to a and epsilon
        a, eps = change_vars(p, forward=False)[:, 0]

        N = 100
        h = 1. / N
        s_eval = (np.arange(0, np.ceil(1. / (vmin * h))) + 1) * h

        # Define initial conditions for solve_pde
        y = np.linspace(0, 1, num=N + 1)
        u_init = delta_h(y[1:], h)
        p0 = np.append(u_init, np.zeros(4 * N))

        u1_bdy = solve_pde(s_eval, p0, h, eps1=eps, eps2=np.inf, a=a,
                           b=1 - a, c=1, d=0, scheme='up')[3]
        cdf = np.interp(1. / v, s_eval, u1_bdy)
        return -np.sum(np.log(cdf))

    if initial_guess is None:
        v_bar = np.mean(vels)
        s2 = np.std(vels) ** 2
        a0 = v_bar - s2 / (2 * v_bar)
        e0 = s2 / (2 * v_bar ** 2 * (1 - v_bar))
        s0 = (2 * a0 - 1) / (2 * a0 * (1 - a0))
        le0 = np.log(e0)
        initial_guess = np.array([s0, le0])

    v = np.array(vels)
    vmin = np.amin(v)
    # It may not be necessary to fit the reduced model to the data
    sol1 = minimize(reduced_objective, initial_guess, args=(v,))
    sol2 = minimize(full_objective, sol1.x, args=(v, vmin))
    # sol_test = minimize(full_objective, initial_guess)

    # Don't need to print out the number of function evaluations
    print(sol1.nfev)
    print(sol2.nfev)
    # print(sol_test.nfev)

    return sol1.x, sol2.x


def boot_trial(vels, initial_guess=None):
    """Run a single bootstrap trial

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    initial_guess : array_like or None, optional
        Initial guess for the minimization procedure. If None, then
        infer a reasonable starting guess from the mean and standard
        deviation of the data

    Returns
    -------
    reduced_trial : ndarray
        Bootstrap ML parameters of the reduced model
    full_trial : ndarray
        Bootstrap ML parameters of the full PDE model
    """
    np.random.seed()
    new_data = np.random.choice(vels, size=len(vels))
    reduced_trial, full_trial = fit_models(new_data,
                                           initial_guess=initial_guess)
    return np.stack([reduced_trial, full_trial], axis=-1)


def bootstrap(vels, boot_trials=10, proc=1):
    """Run a bootstrapping procedure on average velocity data

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    boot_trials : int
        Number of boostrap trials to run
    proc : int
        Number of parallel processes to run

    Returns
    -------
    ndarray
        2x2xboot_trials array of parameter estimates for each bootstrap
        trial
    """
    import multiprocessing as mp
    pool = mp.Pool(processes=proc)
    result = [pool.apply_async(boot_trial, (vels,))
              for _ in range(boot_trials)]

    result = [res.get() for res in result]
    parameter_trials = [change_vars(res, forward=False) for res in result]
    parameter_trials = np.stack(parameter_trials, axis=-1)

    return parameter_trials


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

    # sol = solve_pde(s_eval, p0, h, eps1, eps2, a, b, c, d, scheme)[3]
    # full_model = interp1d(1/s_eval[1:], sol[1:] * s_eval[1:]**2,
    #                       bounds_error=False, fill_value=0)

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
