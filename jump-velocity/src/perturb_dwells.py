import numpy as np
import matplotlib.pyplot as plt
from step_fit_modified import fit_parameters, load_data


def main(filename):
    # np.seterr(all='raise')
    L, V, nd_dwells, nd_steps, small_step_thresh = load_data(filename)
    # Run fits on the unperturbed data, to get reference
    a_orig, conv_orig, beta_orig, d_orig, dwell_res, et_orig, g_orig, k_orig, theta_orig = (
        fit_parameters(L, nd_steps, nd_dwells, small_step_thresh))
    print('alpha = {}, gamma = {}, delta = {}, eta = {}'
          .format(a_orig, g_orig, d_orig, et_orig))
    print(dwell_res.fun)

    # Perturb dwells first (by 0.1 s)
    n_trials = 1024
    perturbed_dwell_estimates = [
        perturbation_dwell_trial(L, V, np.copy(nd_dwells), np.copy(nd_steps),
                                 small_step_thresh, count, n_trials)
        for count in range(n_trials)
    ]

    a_est, g_est = zip(*perturbed_dwell_estimates)
    plt.hist(a_est, density=True)
    plt.axvline(a_orig, c='k')
    plt.title('$\\alpha$ estimates')
    plt.show()

    plt.hist(g_est, density=True)
    plt.axvline(g_orig, c='k')
    plt.title('$\\gamma$ estimates')
    plt.show()

    # Next try perturbing steps (by 0.1 microns) (best to stick with the step
    #                                             size estimates)
    perturbed_step_estimates = [
        perturbation_step_trial(L, V, np.copy(nd_dwells), np.copy(nd_steps),
                                small_step_thresh, count, n_trials)
        for count in range(n_trials)
    ]

    estimates_step = zip(*perturbed_step_estimates)
    originals = [conv_orig, beta_orig, k_orig, theta_orig, a_orig, g_orig]
    titles = [
        '$\\chi$ estimates', '$\\beta$ estimates', '$k$ estimates',
        '$\\theta$ estimates', '$\\alpha$ estimates', '$\\gamma$ estimates'
    ]
    for i, est in enumerate(estimates_step):
        plt.hist(est, density=True)
        plt.axvline(originals[i], c='k')
        plt.title(titles[i])
        plt.show()

    # Finally try both
    perturbed_both_estimates = [
        perturbation_both_trial(L, V, np.copy(nd_dwells), np.copy(nd_steps),
                                small_step_thresh, count, n_trials)
        for count in range(n_trials)
    ]

    estimates_both = zip(*perturbed_both_estimates)
    for i, est in enumerate(estimates_both):
        est = np.array(est)
        plt.hist(est[np.isfinite(est)], density=True)
        plt.axvline(originals[i], c='k')
        plt.title(titles[i])
        plt.show()


def perturbation_dwell_trial(L, V, nd_dwells, nd_steps, small_step_thresh,
                             count, n_trials):
    epsilon = np.random.normal(scale=0.1 * V / L, size=nd_dwells.shape)
    nd_dwells += epsilon
    nd_dwells[nd_dwells < 0] = -nd_dwells[nd_dwells < 0]
    a, conv, beta, d, dwell_res, et, g, k, theta = (
        fit_parameters(L, nd_steps, nd_dwells, small_step_thresh))
    print('Completed {} of {} trials'.format(count+1, n_trials))
    return a, g


def perturbation_step_trial(L, V, nd_dwells, nd_steps, small_step_thresh,
                            count, n_trials):
    epsilon = np.random.normal(scale=0.1 / L, size=nd_steps.shape)
    nd_steps += epsilon
    nd_steps[nd_steps < 0] = -nd_steps[nd_steps < 0]
    a, conv, beta, d, dwell_res, et, g, k, theta = (
        fit_parameters(L, nd_steps, nd_dwells, small_step_thresh))
    print('Completed {} of {} trials'.format(count+1, n_trials))
    return conv, beta, k, theta


def perturbation_both_trial(L, V, nd_dwells, nd_steps, small_step_thresh,
                            count, n_trials):
    epsilon_dwells = np.random.normal(scale=0.1 * V / L, size=nd_dwells.shape)
    nd_dwells += epsilon_dwells
    nd_dwells[nd_dwells < 0] = -nd_dwells[nd_dwells < 0]

    epsilon_steps = np.random.normal(scale=0.1 / L, size=nd_steps.shape)
    nd_steps += epsilon_steps
    nd_steps[nd_steps < 0] = -nd_steps[nd_steps < 0]

    a, conv, beta, d, dwell_res, et, g, k, theta = (
        fit_parameters(L, nd_steps, nd_dwells, small_step_thresh))
    print('Completed {} of {} trials'.format(count+1, n_trials))
    return conv, beta, k, theta, a, g


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    filename = raw_input('Enter the filename: ')

    main(filename)
