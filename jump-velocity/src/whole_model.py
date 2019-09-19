import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from mle import fit_models, change_vars, delta_h, solve_pde


def construct_dwell_fun(d, a, c, eps1, eps2):
    rxn_matrix = np.array([[-a/eps1 - (1-c)/eps2, 0, c/eps2],
                           [0, -(1-a)/eps1 - c/eps2, a/eps1],
                           [(1-c)/eps2, (1-a)/eps1, -a/eps1 - c/eps2]])

    def fun(y, t):
        return np.dot(rxn_matrix, y)

    total_rate = (1-a)/eps1 + (1-c)/eps2
    ic = np.array([(1-a)/eps1/total_rate, (1-c)/eps2/total_rate, 0])

    result = odeint(fun, ic, d)

    return a/eps1*result[:, 0] + c/eps2*result[:, 1]


def main(filename):
    sim_dir = 'dat-files/simulations/'
    steps = np.loadtxt(sim_dir + filename + '-step.dat')
    vels = np.loadtxt(sim_dir + filename + '-vel.dat')
    dwell = np.loadtxt(sim_dir + filename + '-dwell.dat')

    V, L = 1., 1.
    T = L/V

    nd_steps = steps/L
    nd_vels = vels/V
    nd_dwell = dwell/(1*T)

    fit = fit_models(nd_vels, two_par=False, N_obj=32)
    print('Done fitting')
    a_fit, eps1_fit = change_vars(fit[:2], forward=False)[:, 0]
    c_fit, eps2_fit = change_vars(fit[2:], forward=False)[:, 0]

    # fit = fit_models(nd_vels, two_par=True, N_obj=128)[1]
    # a_fit, eps1_fit = change_vars(fit, forward=False)[:, 0]
    # c_fit, eps2_fit = 0.5, np.inf

    # Define numerical parameters to find the PDE at the ML estimates
    N = 256
    s_max = 50
    s_eval = np.linspace(0, s_max, num=s_max * N + 1)
    h = 1. / N
    scheme = None
    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N))

    sol_fit = solve_pde(s_eval, p0, h, eps1_fit, eps2_fit, a_fit,
                        1 - a_fit, c_fit, 1 - c_fit, scheme)[3]

    full_model = interp1d(1 / s_eval[1:], sol_fit[1:]
                          / (1 - np.exp((a_fit - 1) / eps1_fit
                                        + (c_fit - 1) / eps2_fit))
                          * s_eval[1:] ** 2, bounds_error=False, fill_value=0)
    plt.hist(nd_vels, density=True)
    v = np.linspace(0, np.amax(nd_vels), num=200)
    plt.plot(v, full_model(v))
    plt.show()

    plt.hist(nd_steps, density=True)
    s = np.linspace(0, np.amax(nd_steps), num=200)
    tot_bind = (1-a_fit)/eps1_fit + (1-c_fit)/eps2_fit
    plt.plot(s, tot_bind*np.exp(-tot_bind*s))
    plt.show()

    x_cdf = np.sort(nd_steps)
    x_cdf = np.insert(x_cdf, 0, 0)
    y_cdf = np.linspace(0, 1, len(x_cdf))
    plt.step(x_cdf, y_cdf, where='post')
    plt.plot(s, 1 - np.exp(-tot_bind*s))
    plt.show()

    plt.semilogy(x_cdf, 1 - y_cdf)
    plt.semilogy(s, np.exp(-tot_bind*s))
    plt.show()

    plt.hist(nd_dwell, density=True)
    d = np.linspace(0, np.amax(nd_dwell), num=200)
    dwell_model = construct_dwell_fun(d, a_fit, c_fit, eps1_fit, eps2_fit)
    plt.plot(d, dwell_model)
    plt.show()

    print(a_fit, eps1_fit, c_fit, eps2_fit)


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    main(filename)
