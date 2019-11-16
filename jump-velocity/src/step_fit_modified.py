import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import chisquare
import matplotlib.pyplot as plt


def main(filename):
    sim_dir = 'dat-files/simulations/'
    steps = np.loadtxt(sim_dir + filename + '-step.dat')

    def pdf_fun(x, pars):
        conv, beta, lam = pars
        return (conv * beta * np.exp(-beta * x)
                + (1 - conv) * lam * np.exp(-lam * x))

    def objective(pars):
        pdf = lambda x: pdf_fun(x, pars)
        return -np.sum(np.log(pdf(steps)))

    initial_guess = np.array([.5, 10., 100.])  # Change to estimate with MoM
    res = minimize(objective, x0=initial_guess,
                   bounds=[(0, 1), (0, None), (0, None)])
    conv, beta, lam = res.x
    print(conv, beta, lam)

    def pdf(x):
        return pdf_fun(x, np.array([conv, beta, lam]))

    def icdf(t):
        return conv*np.exp(-beta*t) + (1 - conv)*np.exp(-lam*t)

    # Find goodness-of-fit
    num_bins = steps.size//5
    si = np.zeros(shape=num_bins+1)
    for i in range(num_bins-1):
        si[i+1] = brentq(lambda s: icdf(s) - (num_bins - (i + 1.))/num_bins,
                         si[i], np.amax(steps))
    si[-1] = np.inf

    f_obs = np.histogram(steps, si)[0]
    print(chisquare(f_obs, ddof=3))

    t = np.linspace(0, np.amax(steps), num=256)

    plt.hist(steps, density=True)
    plt.plot(t, pdf(t))
    plt.show()

    plt.plot(t, 1 - icdf(t))
    plt.step(np.append(0, np.sort(steps)),
             np.append(0, (np.arange(0, steps.size) + 1.)/steps.size),
             where='post')
    # for s in si[:-1]:
    #     plt.axvline(s, color='k')
    plt.show()

    est_dir = 'dat-files/ml-estimates/'
    save_array = np.zeros(shape=(t.size, 3))
    save_array[:, 0] = t
    save_array[:, 1] = pdf(t)
    save_array[:, 2] = 1 - icdf(t)

    np.savetxt(est_dir + filename + '-dexp.dat', save_array)


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    filename = raw_input('Enter the filename: ')

    main(filename)
