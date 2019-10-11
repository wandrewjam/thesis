import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import brentq
from mle import fit_models, change_vars, delta_h, solve_pde
from generate_test import multiple_experiments


def construct_dwell_fun(d, a, c, eps1, eps2):
    rxn_matrix = np.array([[-a/eps1 - (1-c)/eps2, 0, c/eps2],
                           [0, -(1-a)/eps1 - c/eps2, a/eps1],
                           [(1-c)/eps2, (1-a)/eps1, -a/eps1 - c/eps2]])

    def fun(y, t):
        return np.dot(rxn_matrix, y)

    total_rate = (1-a)/eps1 + (1-c)/eps2
    ic = np.array([(1-a)/eps1/total_rate, (1-c)/eps2/total_rate, 0])

    result = odeint(fun, ic, d)

    return a/eps1*result[:, 0] + c/eps2*result[:, 1], 1 - np.sum(result, axis=1)


def main(filename):
    sim_dir = 'dat-files/simulations/'
    steps = np.loadtxt(sim_dir + filename + '-step.dat')
    vels = np.loadtxt(sim_dir + filename + '-vel.dat')
    dwell = np.loadtxt(sim_dir + filename + '-dwell.dat')

    V, L = 1., 1.
    # V, L = 60., 80.
    T = L/V

    nd_steps = steps/L
    nd_vels = vels/V
    nd_dwell = dwell/(1*T)
    # nd_dwell = dwell/(1000*T)

    # fit = fit_models(nd_vels, two_par=False, N_obj=32)
    # print('Done fitting')
    # a_fit, eps1_fit = change_vars(fit[:2], forward=False)[:, 0]
    # c_fit, eps2_fit = change_vars(fit[2:], forward=False)[:, 0]

    # Fit to step size first
    # def objective(eps1):
    #     # What if I fix eps1 and only fit a?
    #     a = .5
    #     test_steps = multiple_experiments(a, .5, eps1, np.inf, 2**10)[1]
    #     print(np.mean(test_steps) - np.mean(nd_steps))
    #     return np.mean(test_steps) - np.mean(nd_steps)
    #
    # res = brentq(objective, 0.01, 10)
    # print('The result is')
    # print(res)
    # fit = fit_models(nd_vels, two_par=True, N_obj=128)[1]
    # a_fit, eps1_fit = change_vars(fit, forward=False)[:, 0]
    # c_fit, eps2_fit = 0.5, np.inf

    # a_fit, eps1_fit = .5, .1
    # c_fit, eps2_fit = .2, np.inf
    # fit1, fit2 = change_vars(np.array([.5, res]), forward=True)[:, 0]
    rate_fit = float(nd_dwell.size)/nd_vels.size
    print(rate_fit)
    c_fit, eps2_fit = 0.2, np.inf
    #
    constraints = {'type': 'eq', 'fun': lambda x: x[0]/x[1] - rate_fit}
    vel_fit = fit_models(nd_vels, two_par=True, N_obj=128,
                         constraints=constraints,
                         initial_guess=np.array([.3, .7/rate_fit]))[1]
    a_fit, eps1_fit = vel_fit

    # a_fit, eps1_fit = 0.5, 0.8

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
    plt.legend(['Data', 'Model'])
    plt.show()

    plt.hist(nd_steps, density=True)
    s = np.linspace(0, np.amax(nd_steps), num=200)
    tot_bind = (1-a_fit)/eps1_fit + (1-c_fit)/eps2_fit
    plt.plot(s, tot_bind*np.exp(-tot_bind*s))
    plt.legend(['Data', 'Model'])
    plt.show()

    x_cdf = np.sort(nd_steps)
    x_cdf = np.insert(x_cdf, 0, 0)
    y_cdf = np.linspace(0, 1, len(x_cdf))
    plt.step(x_cdf, y_cdf, where='post')
    plt.plot(s, 1 - np.exp(-tot_bind*s))
    plt.legend(['Data', 'Model'])
    plt.show()

    plt.semilogy(x_cdf, 1 - y_cdf)
    plt.semilogy(s, np.exp(-tot_bind*s))
    plt.legend(['Data', 'Model'])
    plt.show()

    plt.hist(nd_dwell, density=True)
    d = np.linspace(0, np.amax(nd_dwell), num=200)
    dwell_model, dwell_cdf = construct_dwell_fun(d, a_fit, c_fit, eps1_fit, eps2_fit)
    plt.plot(d, dwell_model)
    plt.legend(['Data', 'Model'])
    plt.show()

    x_cdf = np.sort(nd_dwell)
    x_cdf = np.insert(x_cdf, 0, 0)
    y_cdf = np.linspace(0, 1, len(x_cdf))
    plt.step(x_cdf, y_cdf, where='post')
    plt.plot(d, dwell_cdf)
    plt.legend(['Data', 'Model'])
    plt.show()

    print(a_fit, eps1_fit, c_fit, eps2_fit)


if __name__ == '__main__':
    import os
    import sys
    # filename = sys.argv[1]
    filename = raw_input('Enter filename: ')
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    main(filename)
