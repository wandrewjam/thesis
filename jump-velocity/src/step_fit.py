import numpy as np
from scipy.integrate import cumtrapz
from scipy.optimize import brentq, minimize
from scipy.stats import anderson, chisquare
import matplotlib.pyplot as plt


def main(filename, model='simple'):
    sim_dir = 'dat-files/simulations/'
    steps = np.loadtxt(sim_dir + filename + '-step.dat')

    if model == 'simple':
        scale_mle = np.mean(steps)
        print(anderson(steps, dist='expon'))
        print(scale_mle)

        def pdf(t):
            return 1/scale_mle * np.exp(-t/scale_mle)

    elif model == 'double':
        def objective(pars):
            a, lam1, lam2 = pars
            return -np.sum(np.log(a*lam1*np.exp(-lam1*steps)
                                  + (1-a)*lam2*np.exp(-lam2*steps)))
        initial_guess = np.array([0.6, 1., .2])
        res = minimize(objective, x0=initial_guess,
                       bounds=[(0, 1), (0, None), (0, None)])
        a, lam1, lam2 = res.x
        print(a, lam1, lam2)

        def pdf(t):
            return a*lam1*np.exp(-lam1*t) + (1 - a)*lam2*np.exp(-lam2*t)

        def icdf(t):
            return a*np.exp(-lam1*t) + (1 - a)*np.exp(-lam2*t)

        ## Find goodness-of-fit
        num_bins = steps.size//5
        si = np.zeros(shape=num_bins+1)
        for i in range(num_bins-1):
            si[i+1] = brentq(lambda s: icdf(s) - (num_bins - (i + 1.))/num_bins,
                             si[i], np.amax(steps))
        si[-1] = np.inf

        f_obs = np.histogram(steps, si)[0]
        print(f_obs)
        print(chisquare(f_obs, ddof=3))

    else:
        raise ValueError('variable model must be simple or double')

    t = np.linspace(0, np.amax(steps), num=100)

    plt.hist(steps, density=True)
    plt.plot(t, pdf(t))
    plt.show()

    plt.plot(t, cumtrapz(pdf(t), t, initial=0))
    plt.step(np.append(0, np.sort(steps)),
             np.append(0, (np.arange(0, steps.size) + 1.)/steps.size),
             where='post')
    for s in si[:-1]:
        plt.axvline(s, color='k')
    plt.show()

    est_dir = 'dat-files/ml-estimates/'
    save_array = np.zeros(shape=(t.size, 3))
    save_array[:, 0] = t
    save_array[:, 1] = pdf(t)
    save_array[:, 2] = cumtrapz(pdf(t), t, initial=0)

    if model == 'simple':
        np.savetxt(est_dir + filename + '-exp.dat', save_array)
    elif model == 'double':
        np.savetxt(est_dir + filename + '-dexp.dat', save_array)


if __name__ == '__main__':
    import os
    import sys
    # filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    filename = raw_input('Enter the filename: ')
    model = raw_input('Enter the model you\'d like to run: ')

    main(filename, model)
