import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from jv import solve_pde, delta_h


def change_vars(p, forward=True):
    """Convert between model parameters and fitting parameters

    Parameters
    ----------
    p : array_like
        Parameter values to convert
    forward : bool, optional
        If True, converts model parameters to fitting parameters. If
        False, converts fitting parameters to model parameters.

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


def save_data(reduced, full_model, filename):
    """

    Parameters
    ----------
    filename
    reduced
    full_model

    """
    save_array = np.zeros(shape=(500, 5))
    x_plot = np.linspace(0, 1.5, num=501)[1:]
    save_array[:, 0] = x_plot
    save_array[:, 1] = reduced(x_plot)
    save_array[:, 2] = 1 + cumtrapz(reduced(x_plot[::-1]), x_plot[::-1],
                                    initial=0)[::-1]
    save_array[:, 3] = full_model(x_plot)
    save_array[:, 4] = 1 + cumtrapz(full_model(x_plot[::-1]), x_plot[::-1],
                                    initial=0)[::-1]

    est_dir = 'dat-files/ml-estimates/'
    np.savetxt(est_dir + filename + '-est.dat', save_array)


def fit_models(vels, initial_guess=None):
    """Fits the adiabatic reduction and full PDE model to data

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    initial_guess : array_like or None, optional
        Initial guess for the minimization procedure. If None, then
        infer a reasonable starting guess from the mean and standard
        deviation of the data using a method of moments estimate.

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

    def reduced_objective(p, vels, vectorized=False):
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

    sol1 = minimize(reduced_objective, initial_guess, args=(v,))
    sol2 = minimize(full_objective, sol1.x, args=(v, vmin))

    return sol1.x, sol2.x


def main(filename):
    sim_dir = 'dat-files/simulations/'
    vels = np.loadtxt(sim_dir + filename + '-sim.dat')

    def reduced(v, a_reduced, eps_reduced):
        b_reduced = 1 - a_reduced
        return (1 / np.sqrt(4 * np.pi * eps_reduced
                            * a_reduced * b_reduced * v ** 3)
                * (a_reduced + (v - a_reduced) / 2)
                * np.exp(-(v - a_reduced) ** 2 / (4 * eps_reduced * a_reduced
                                                  * b_reduced * v)))

    reduced_fit, full_fit = fit_models(vels)
    a_rfit, eps_rfit = change_vars(reduced_fit, forward=False)[:, 0]
    a_ffit, eps_ffit = change_vars(full_fit, forward=False)[:, 0]

    # Define numerical parameters to find the PDE at the ML estimates
    N = 1000
    s_max = 50
    s_eval = np.linspace(0, s_max, num=s_max * N + 1)
    h = 1. / N
    scheme = None
    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N))

    sol_fit = solve_pde(s_eval, p0, h, eps_ffit, np.inf, a_ffit, 1 - a_ffit,
                        1, 0, scheme)[3]
    full_model = interp1d(1 / s_eval[1:], sol_fit[1:] * s_eval[1:] ** 2,
                          bounds_error=False, fill_value=0)

    save_data(lambda v: reduced(v, a_rfit, eps_rfit), full_model, filename)
    # plot_experiments(vels, reduced=lambda v: reduced(v, a_rfit, eps_rfit),
    #                  full_model=full_model)

    # parameter_trials = bootstrap(vels, boot_trials=4, proc=2)
    #
    # print(np.array([[a_rfit, a_ffit], [eps_rfit, eps_ffit]]))
    # print(parameter_trials)
    # print(parameter_trials.shape)
    # print(np.percentile(parameter_trials, q=(50*alpha, 100 - 50*alpha),
    #                     axis=-1))


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    main(filename)