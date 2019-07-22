import numpy as np

from src.jv import delta_h, solve_pde


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
