import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import expon, gamma
from step_fit_modified import fit_trunc_gamma, dwell_pdf_fun, dwell_obj
from simulate_modified import modified_experiment


def main(filename):
    sim_dir = 'dat-files/simulations/'
    steps = np.loadtxt(sim_dir + filename + '-step.dat')
    dwells = np.loadtxt(sim_dir + filename + '-dwell.dat')
    avg_vels = np.loadtxt(sim_dir + filename + '-vel.dat')

    L = 100
    nd_steps = steps / L
    small_step_thresh = 2. / L

    # Fit small steps
    small_steps = nd_steps[nd_steps < small_step_thresh]
    k, theta = fit_trunc_gamma(small_steps, b=small_step_thresh)

    # Then fit all steps
    def step_pdf_fun(x, pars):
        conv, beta = pars
        return (conv * expon.pdf(x, scale=1./beta)
                + (1 - conv) * gamma.pdf(x, k, scale=theta))

    def objective(pars):
        pdf = lambda x: step_pdf_fun(x, pars)
        return -np.sum(np.log(pdf(nd_steps)))

    initial_guess = np.array([.5, .2 * L])  # Change to estimate with MoM if necessary
    res = minimize(objective, x0=initial_guess,
                   bounds=[(0, 1), (0.01, None)])
    conv, beta = res.x

    # Now fit dwells subject to the constraints that we already fitted step parameters
    # V = np.amax(avg_vels)
    V = 60.
    dwells = np.sort(dwells)
    nd_dwells = dwells * V / L

    def const_fun(fit_pars):
        a, g, d, et = np.exp(fit_pars)
        return a * (g + et) / (a * (g + et) + d * et) - conv

    def const_jac(fit_pars):
        a, g, d, et = np.exp(fit_pars)
        # a, g, d, et = pars
        coeff = (a * (g + et) + d * et)**2
        arr = np.array([d*et*(g + et), a*d*et, -a*et*(g + et), -a*et*d])
        return arr / coeff

    const1 = {'type': 'eq', 'fun': const_fun, 'jac': const_jac}
    const2 = {'type': 'eq', 'fun': lambda p: p[0] - p[3],
              'jac': lambda p: np.array([1, 0, 0, -1])}

    p0 = 0 * np.ones(shape=(4,))
    p0[2] = 0
    p0[1] = np.floor(np.log(conv/(1 - conv)) + p0[2])
    p0[0] = np.log(
        (np.exp(p0[1]) * (1 - conv) - conv * np.exp(p0[2])) / (conv - 1)
    )
    p0[3] = p0[0]

    dwell_res = minimize(lambda p: dwell_obj(p, nd_dwells), p0,
                         constraints=[const1, const2])

    def dwell_pdf(t):
        return dwell_pdf_fun(t, np.exp(dwell_res.x))

    a, g, d, et = np.exp(dwell_res.x)

    num_expts = 1024
    results = [
        modified_experiment(a, beta, g, d, et, 'gamma', k=k, theta=theta)
        for _ in range(num_expts)
    ]

    vels = list()
    for res in results:
        vels.append((res[0][-2] - res[0][1])/(res[1][-2] - res[1][1]) * V)

    vels = np.array(vels)

    opacity = 0.8
    fig, ax = plt.subplots(ncols=2, figsize=[10., 4.], sharex='all')
    ax[0].hist(avg_vels, density=True, alpha=opacity)

    vels2 = np.loadtxt(os.path.expanduser('~/thesis/vlado-data/ccp-vels.dat'))
    ax[0].hist(vels2, density=True, alpha=opacity)
    ax[0].hist(vels, density=True, alpha=opacity)
    ax[0].legend(['Original avg. vels.', 'Modified avg. vels.',
                  'Simulated velocities'])
    ax[0].set_xlim(0, 10)
    ax[0].set_xlabel('Average velocity $(\\mu m / s)$')
    ax[0].set_ylabel('Probability density')

    ax[1].step(np.append(0, np.sort(avg_vels)),
               np.append(0, (np.arange(0, avg_vels.size) + 1.)/avg_vels.size),
               where='post')
    ax[1].step(np.append(0, np.sort(vels2)),
               np.append(0, (np.arange(0, vels2.size) + 1.)/vels2.size),
               where='post')
    ax[1].step(np.append(0, np.sort(vels)),
               np.append(0, (np.arange(0, vels.size)
                             + 1.)/vels.size),
               where='post')
    ax[1].legend(['Original avg. vels.', 'Modified avg. vels.',
                  'Simulated velocities'])
    ax[1].set_xlabel('Average velocity $(\\mu m / s)$')
    ax[1].set_ylabel('Cumulative probability')
    plt.tight_layout()
    plt.show()

    print(np.mean(avg_vels), np.mean(vels2), np.mean(vels))


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    filename = raw_input('Enter the filename: ')

    main(filename)
