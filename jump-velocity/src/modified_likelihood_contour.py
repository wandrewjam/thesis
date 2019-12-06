import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import expon, gamma
from step_fit_modified import dwell_obj, fit_trunc_gamma


def main(filename):
    simplified = True
    sim_dir = 'dat-files/simulations/'
    steps = np.loadtxt(sim_dir + filename + '-step.dat')
    small_step_thresh = 2

    # Fit small steps
    small_steps = steps[steps < small_step_thresh]
    k, theta = fit_trunc_gamma(small_steps, b=small_step_thresh)

    # Then fit all steps
    def step_pdf_fun(x, pars):
        conv, beta = pars
        return (conv * expon.pdf(x, scale=1./beta)
                + (1 - conv) * gamma.pdf(x, k, scale=theta))

    def objective(pars):
        pdf = lambda x: step_pdf_fun(x, pars)
        return -np.sum(np.log(pdf(steps)))

    initial_guess = np.array([.5, .2])  # Change to estimate with MoM if necessary
    res = minimize(objective, x0=initial_guess,
                   bounds=[(0, 1), (0.01, None)])
    conv, beta = res.x

    dwells = np.loadtxt(sim_dir + filename + '-dwell.dat')
    dwells = np.sort(dwells)

    # Define range of parameters for which to plot the likelihood function

    if simplified:
        a_bounds = [-2, 0]
        g_bounds = [-5, 5]
        ln_a_range = np.linspace(*a_bounds)
        ln_g_range = np.linspace(*g_bounds)

        likelihoods = np.zeros(shape=ln_a_range.shape + ln_g_range.shape)

        for j, ln_a in enumerate(ln_a_range):
            for i, ln_g in enumerate(ln_g_range):
                ln_et = ln_a
                ln_d = np.log((np.exp(ln_a) + np.exp(ln_g)) * (1 - conv) / conv)

                pars = np.array([ln_a, ln_g, ln_d, ln_et])
                likelihoods[i, j] = dwell_obj(pars, dwells)

        extent = np.append(a_bounds, g_bounds)
        plt.imshow(likelihoods, extent=extent, aspect='auto')
        plt.title('Heatmap of $-\\ln(L)$')
        plt.xlabel('$\\ln(\\alpha)$')
        plt.ylabel('$\\ln(\\gamma)$')
        plt.colorbar()
        plt.show()
    else:
        a_bounds = [-2, 0]
        g_bounds = [-5, 5]
        et_bounds = [-3, 3]
        ln_a_range = np.linspace(*a_bounds)
        ln_g_range = np.linspace(*g_bounds)
        ln_et_range = np.linspace(*et_bounds)

        likelihoods = np.zeros(shape=ln_a_range.shape + ln_g_range.shape
                               + ln_et_range.shape)

        for k, ln_a in enumerate(ln_a_range):
            for j, ln_g in enumerate(ln_g_range):
                for i, ln_et in enumerate(ln_et_range):
                    ln_d = np.log((np.exp(ln_a) * (np.exp(ln_g)
                                                   + np.exp(ln_et))
                                   * (1 - conv)) / (np.exp(ln_et) * conv))

                    pars = np.array([ln_a, ln_g, ln_d, ln_et])
                    likelihoods[i, j, k] = dwell_obj(pars, dwells)

        extent = np.append(a_bounds, g_bounds)
        ii = 22
        plt.imshow(likelihoods[ii, :, :], extent=extent, aspect='auto')
        plt.title('Heatmap of $-\\ln(L)$, with $\\eta = {}$'
                  .format(np.exp(ln_et_range[ii])))
        plt.xlabel('$\\ln(\\alpha)$')
        plt.ylabel('$\\ln(\\gamma)$')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    filename = raw_input('Enter the filename: ')

    main(filename)
