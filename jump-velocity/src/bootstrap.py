import numpy as np
from timeit import default_timer as timer
from mle import fit_models, change_vars


def boot_trial(vels, initial_guess=None, k=None, trials=None):
    """Run a single bootstrap trial

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    initial_guess : array_like or None, optional
        Initial guess for the minimization procedure. If None, then
        infer a reasonable starting guess from the mean and standard
        deviation of the data
    k : int
    trials : int

    Returns
    -------
    ndarray
        (2 x 2) array of ML estimates of the reduced and full models. The
        elements of the array are:
            (parameters from fitting the reduced model)
            (0, 0) - a_rfit
            (1, 0) - eps_rfit
            (parameters from fitting the full model)
            (0, 1) - a_ffit
            (0, 2) - eps_ffit
    """
    start = timer()
    np.random.seed()
    new_data = np.random.choice(vels, size=len(vels))
    reduced_trial, full_trial = fit_models(new_data,
                                           initial_guess=initial_guess)
    end = timer()

    if k is not None and trials is not None:
        print('Completed {:d} of {:d} bootstrap trials. This run took {:g} '
              'seconds.'.format(k+1, trials,
                                end-start))
    elif k is not None:
        print('Completed {:d} bootstrap trials so far. This run took {:g} '
              'seconds.'.format(k+1, end-start))
    elif trials is not None:
        print('Completed one of {:d} bootstrap trials. This run took {:g} '
              'seconds.'.format(trials, end-start))
    else:
        print('Completed one bootstrap trial. This run took {:g} seconds.'
              .format(end-start))
    return np.stack((reduced_trial, full_trial), axis=-1)


def bootstrap(vels, boot_trials=10, proc=1, initial_guess=None):
    """Run a bootstrapping procedure on average velocity data

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    boot_trials : int
        Number of boostrap trials to run
    proc : int
        Number of parallel processes to run

    Returns
    -------
    ndarray
        (4 x boot_trials) array of parameter estimates from the
        bootstrapping procedure. Each column of the array contains the
        ML estimates for a single bootstrap trial in the following
        order:
            (parameters from fitting the reduced model)
            0 - a_rfit
            1 - eps_rfit
            (parameters from fitting the full model)
            2 - a_ffit
            3 - eps_ffit
    """
    import multiprocessing as mp
    pool = mp.Pool(processes=proc)
    result = [pool.apply_async(boot_trial,
                               (vels, initial_guess, k, boot_trials))
              for k in range(boot_trials)]

    result = [res.get() for res in result]
    parameter_trials = [change_vars(res, forward=False) for res in result]
    parameter_trials = np.stack(parameter_trials, axis=-1)

    return np.reshape(parameter_trials, newshape=(4, -1), order='F')


def main(filename, boot_trials, proc):
    sim_dir = 'dat-files/simulations/'
    vels = np.loadtxt(sim_dir + filename + '-sim.dat')

    parameter_trials = bootstrap(vels, boot_trials, proc)

    boot_dir = 'dat-files/bootstrap/'
    np.savetxt(boot_dir + filename + '-boot.dat', parameter_trials.T)


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    boot_trials = int(sys.argv[2])
    proc = int(sys.argv[3])
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    main(filename, boot_trials, proc)
