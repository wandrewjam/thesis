import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import chisquare, expon, gamma
from scipy.special import gammainc, gammaln
from scipy.integrate import cumtrapz
from simulate_modified import main as main_sim


def fit_trunc_gamma(data, b=2., initial_guess=None):
    n = data.size

    def objective_fn(pars):
        # Fitting pars are the log values of the model pars
        k_hat, th_hat = pars
        return -((np.exp(k_hat) - 1) * np.sum(np.log(data))
                 - np.exp(-th_hat) * np.sum(data) - n * th_hat * np.exp(k_hat)
                 - n * np.log(gammainc(np.exp(k_hat), b * np.exp(-th_hat)))
                 - n * gammaln(np.exp(k_hat)))

    if initial_guess is None:
        x_bar, s2_bar = np.mean(data), np.var(data)
        p0 = np.array([x_bar**2/s2_bar, s2_bar/x_bar])
    else:
        p0 = initial_guess
    res = minimize(objective_fn, np.log(p0))
    return np.exp(res.x)


def step_pdf_fun(x, k, theta, pars):
    conv, beta = pars
    return (conv * expon.pdf(x, scale=1. / beta)
            + (1 - conv) * gamma.pdf(x, k, scale=theta))


def dwell_pdf_fun(t, pars):
    a, g, d, et = pars

    aa = a + d
    bb = g + et
    dd = np.sqrt((aa - bb)**2 + 4 * g * d)
    l1 = -(aa + bb + dd) / 2
    l2 = (dd - aa - bb) / 2

    pb = ((aa - bb + dd) / (2 * dd) * np.exp(l1 * t)
          + (bb - aa + dd) / (2 * dd) * np.exp(l2 * t))
    pbf = (((aa - bb)**2 - dd**2) / (4 * dd * g) * np.exp(l1 * t)
           + (dd**2 - (aa - bb)**2) / (4 * dd * g) * np.exp(l2 * t))

    return a * pb + et * pbf


def dwell_obj(fit_pars, dwells):
    pars = np.exp(fit_pars)
    g_fun = dwell_pdf_fun(dwells, pars)
    return -np.sum(np.log(g_fun))


def load_data(filename):
    sim_dir = 'dat-files/simulations/'
    L = 100
    steps = np.loadtxt(sim_dir + filename + '-step.dat')
    nd_steps = steps / L
    small_step_thresh = 2. / L
    dwells = np.loadtxt(sim_dir + filename + '-dwell.dat')
    avg_vels = np.loadtxt(sim_dir + filename + '-vel.dat')
    # V = np.amax(avg_vels)
    V = 10
    dwells = np.sort(dwells)
    nd_dwells = dwells * V / L
    return L, V, nd_dwells, nd_steps, small_step_thresh


def fit_parameters(L, nd_steps, nd_dwells, small_step_thresh,
                   print_results=False):
    # The function is fitting the exponential to the short steps and the
    # gamma to the long ones. I could fit the small steps first, get
    # gamma parameters, and then fit the whole thing? Yes, this is now
    # what I am doing.

    # Fit small steps
    small_steps = nd_steps[nd_steps < small_step_thresh]
    k, theta = fit_trunc_gamma(small_steps, b=small_step_thresh)

    # Then fit all steps
    def objective(pars, k, theta):
        pdf = lambda x: step_pdf_fun(x, k, theta, pars)
        return -np.sum(np.log(pdf(nd_steps)))

    # Change to estimate with MoM if necessary
    initial_guess = np.array([.5, .2 * L])
    res = minimize(objective, args=(k, theta), x0=initial_guess,
                   bounds=[(0, 1), (0.01, None)])
    conv, beta = res.x
    if print_results:
        print('chi = {}, beta = {}, k = {}, theta = {}'
              .format(conv, beta, k, theta))

    def const_fun(fit_pars):
        a, g, d, et = np.exp(fit_pars)
        return a * (g + et) / (a * (g + et) + d * et) - conv

    def const_jac(fit_pars):
        a, g, d, et = np.exp(fit_pars)
        # a, g, d, et = pars
        coeff = (a * (g + et) + d * et) ** 2
        arr = np.array([d * et * (g + et), a * d * et,
                        -a * et * (g + et), -a * et * d])
        return arr / coeff

    const1 = {'type': 'eq', 'fun': const_fun, 'jac': const_jac}
    const2 = {'type': 'eq', 'fun': lambda p: p[0] - p[3],
              'jac': lambda p: np.array([1, 0, 0, -1])}
    p0 = 0 * np.ones(shape=(4,))
    p0[2] = 0
    p0[1] = np.floor(np.log(conv / (1 - conv)) + p0[2])
    p0[0] = np.log(
        (np.exp(p0[1]) * (1 - conv) - conv * np.exp(p0[2])) / (conv - 1)
    )
    p0[3] = p0[0]
    dwell_res = minimize(lambda p: dwell_obj(p, nd_dwells), p0,
                         constraints=[const1, const2], bounds=[(None, 5)] * 4)
    a, g, d, et = np.exp(dwell_res.x)
    return a, conv, beta, d, dwell_res, et, g, k, theta


def main(filename):
    L, V, nd_dwells, nd_steps, small_step_thresh = load_data(filename)

    a, conv, beta, d, dwell_res, et, g, k, theta = fit_parameters(L, nd_steps, nd_dwells, small_step_thresh)

    def step_pdf(x):
        return step_pdf_fun(x, k, theta, np.array([conv, beta]))

    def icdf(t):
        return (1 - conv*expon.cdf(t, scale=1./beta)
                - (1 - conv)*gamma.cdf(t, k, scale=theta))

    # Find goodness-of-fit
    num_bins = nd_steps.size//5
    si = np.zeros(shape=num_bins+1)
    for i in range(num_bins-1):
        si[i+1] = brentq(lambda s: icdf(s) - (num_bins - (i + 1.))/num_bins,
                         si[i], np.amax(nd_steps))
    si[-1] = np.inf

    f_obs = np.histogram(nd_steps, si)[0]
    print(chisquare(f_obs, ddof=3))

    t = np.linspace(0, np.amax(nd_steps), num=257)

    plt.hist(nd_steps, density=True)
    plt.plot(t, step_pdf(t))
    plt.show()

    plt.plot(t, 1 - icdf(t))
    plt.step(np.append(0, np.sort(nd_steps)),
             np.append(0, (np.arange(0, nd_steps.size) + 1.)/nd_steps.size),
             where='post')
    # for s in si[:-1]:
    #     plt.axvline(s, color='k')
    plt.show()

    est_dir = 'dat-files/ml-estimates/'
    step_save_array = np.zeros(shape=(t.size, 3))
    step_save_array[:, 0] = t * L
    step_save_array[:, 1] = step_pdf(t) / L
    step_save_array[:, 2] = 1 - icdf(t)

    np.savetxt(est_dir + filename + '-step-dst.dat', step_save_array)

    def dwell_pdf(t):
        return dwell_pdf_fun(t, np.exp(dwell_res.x))
    print('alpha = {}, gamma = {}, delta = {}, eta = {}'.format(a, g, d, et))
    print(dwell_res.fun)

    t_plot = np.linspace(0, nd_dwells[-1], num=257)
    plt.hist(nd_dwells, density=True)
    plt.plot(t_plot, dwell_pdf(t_plot))
    plt.show()

    plt.plot(t_plot, cumtrapz(dwell_pdf(t_plot), t_plot, initial=0))
    plt.step(np.append(0, np.sort(nd_dwells)),
             np.append(0, (np.arange(0, nd_dwells.size) + 1.)/nd_dwells.size),
             where='post')
    plt.show()

    dwell_save_array = np.zeros(shape=(t_plot.size, 3))
    dwell_save_array[:, 0] = t_plot * L / V
    dwell_save_array[:, 1] = dwell_pdf(t_plot) * V / L
    dwell_save_array[:, 2] = cumtrapz(dwell_save_array[:, 1], t_plot * L / V,
                                      initial=0)

    # np.savetxt(est_dir + filename + '-dwell-dst.dat', dwell_save_array)

    # np.savetxt(sim_dir + filename + '-dwell.dat', np.sort(dwells))

    # To-Do: generate trials of average velocity and compare with
    # experimental velocity. I could save the parameter file, and then
    # run the simulation from simulate_modified.

    # Import avg vels

    sim_filename = filename + 'samp'
    # v = 67.9
    # sim_filename = filename + 'samp2'

    # v = 200.
    # sim_filename = filename + 'samp3'

    main_sim(sim_filename, alpha=a, beta=beta, gamma=g, delta=d,
             eta=et, num_expt=2**10, dwell_type='gamma', k=k,
             theta=theta)

    # What else can I do? Decide based on the filename whether it is data or a
    # full simulation, then choose the upper bound for small step sizes


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    filename = raw_input('Enter the filename: ')

    main(filename)
