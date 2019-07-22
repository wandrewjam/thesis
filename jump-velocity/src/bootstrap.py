import numpy as np
from src.mle import fit_models, change_vars


def boot_trial(vels, initial_guess=None):
    """Run a single bootstrap trial

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    initial_guess : array_like or None, optional
        Initial guess for the minimization procedure. If None, then
        infer a reasonable starting guess from the mean and standard
        deviation of the data

    Returns
    -------
    reduced_trial : ndarray
        Bootstrap ML parameters of the reduced model
    full_trial : ndarray
        Bootstrap ML parameters of the full PDE model
    """
    np.random.seed()
    new_data = np.random.choice(vels, size=len(vels))
    reduced_trial, full_trial = fit_models(new_data,
                                           initial_guess=initial_guess)
    return np.stack([reduced_trial, full_trial], axis=-1)


def bootstrap(vels, boot_trials=10, proc=1):
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
        2x2xboot_trials array of parameter estimates for each bootstrap
        trial
    """
    import multiprocessing as mp
    pool = mp.Pool(processes=proc)
    result = [pool.apply_async(boot_trial, (vels,))
              for _ in range(boot_trials)]

    result = [res.get() for res in result]
    parameter_trials = [change_vars(res, forward=False) for res in result]
    parameter_trials = np.stack(parameter_trials, axis=-1)

    return parameter_trials
