import numpy as np
from scipy.stats import expon, gamma, norm, uniform, truncnorm
import os
import sys


if __name__ == '__main__':
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    sim_dir = 'dat-files/simulations/'
    data = np.loadtxt(sim_dir + filename + '-step.dat')

    new_data = data[data < 2.]
    np.savetxt(sim_dir + filename + '-small-step.dat', new_data)

    dst_dir = 'dat-files/distributions/'
    xplot = np.linspace(0, 2, num=201)

    scale = np.mean(new_data)
    e_rv = expon(loc=0, scale=scale)

    # Put in AIC estimation stuff?

    print('Exponential: lam={}'.format(scale))

    e_pdf = e_rv.pdf(xplot)
    e_cdf = e_rv.cdf(xplot)

    loc, scale = uniform.fit(new_data)
    u_rv = uniform(loc=loc, scale=scale)

    print('Uniform: a={}, b={}'.format(loc, loc+scale))

    u_pdf = u_rv.pdf(xplot)
    u_cdf = u_rv.cdf(xplot)

    mu, std = np.mean(new_data), np.std(new_data)

    print('Normal: mu={}, std={}'.format(mu, std))

    n_rv = norm(loc=mu, scale=std)
    n_pdf = n_rv.pdf(xplot)
    n_cdf = n_rv.cdf(xplot)

    g_rv = gamma(a=mu**2/std**2, scale=std**2/mu)
    print('Gamma: theta={}, k={}'.format(std**2/mu, mu**2/std**2))

    g_pdf = g_rv.pdf(xplot)
    g_cdf = g_rv.cdf(xplot)

    save_data = np.array([xplot, e_pdf, e_cdf, u_pdf, u_cdf, n_pdf, n_cdf,
                          g_pdf, g_cdf])
    np.savetxt(dst_dir + filename + '-small-step.dat', save_data.T)
