import numpy as np
from scipy.stats import expon, gamma, norm, uniform, truncnorm
from scipy.special import gammaln
import os
import sys


if __name__ == '__main__':
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    sim_dir = 'dat-files/simulations/'
    # data = np.loadtxt(sim_dir + filename + '-step.dat')
    #
    # new_data = data[data < 2.]
    # np.savetxt(sim_dir + filename + '-small-step.dat', new_data)
    new_data = np.loadtxt(sim_dir + filename + '-small-step.dat')

    n = len(new_data)

    dst_dir = 'dat-files/distributions/'
    xplot = np.linspace(0, 2, num=201)

    scale = np.mean(new_data)
    e_rv = expon(loc=0, scale=scale)

    # Put in AIC estimation stuff?

    e_ll = -n * np.log(scale) - n * scale**(-2)
    e_aic = 2 - 2 * e_ll

    print('Exponential: lam={}. AIC: {}'.format(scale, e_aic))

    e_pdf = e_rv.pdf(xplot)
    e_cdf = e_rv.cdf(xplot)

    loc, scale = 0, 2
    u_rv = uniform(loc=loc, scale=scale)

    u_ll = -n * np.log(scale)
    u_aic = -2 * u_ll

    print('Uniform: a={}, b={}. AIC: {}'.format(loc, loc+scale, u_aic))

    u_pdf = u_rv.pdf(xplot)
    u_cdf = u_rv.cdf(xplot)

    mu, std = np.mean(new_data), np.std(new_data)
    n_aic = (4 + n * np.log(2 * np.pi) + n * np.log(std**2)
             + np.sum((new_data - mu)**2) / std**2)

    print('Normal: mu={}, std={}. AIC: {}'.format(mu, std, n_aic))

    n_rv = norm(loc=mu, scale=std)
    n_pdf = n_rv.pdf(xplot)
    n_cdf = n_rv.cdf(xplot)

    a, scale = gamma.fit(new_data, floc=0)[::2]
    g_rv = gamma(a=a, scale=scale)

    g_ll = ((a - 1) * np.sum(np.log(new_data)) - n * mu / scale
            - n * a * np.log(scale) - n * gammaln(a))
    g_aic = 4 - 2 * g_ll

    print('Gamma: theta={}, k={}. AIC: {}'.format(scale, a, g_aic))

    g_pdf = g_rv.pdf(xplot)
    g_cdf = g_rv.cdf(xplot)

    save_data = np.array([xplot, e_pdf, e_cdf, u_pdf, u_cdf, n_pdf, n_cdf,
                          g_pdf, g_cdf])
    np.savetxt(dst_dir + filename + '-small-step.dat', save_data.T)
