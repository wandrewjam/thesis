import numpy as np
from simulate import read_parameter_file, multiple_experiments
from mle import fit_models, change_vars
from timeit import default_timer as timer


def mle_trial_wrapper(vels, two_par, N_obj, k=None, trials=None):
    print('Beginning trial {:d} of {:d}'.format(k+1, trials))
    start = timer()
    result = fit_models(vels=vels, two_par=two_par, N_obj=N_obj)
    end = timer()

    if k is not None and trials is not None:
        print('Completed {:d} of {:d} trials. This run took {:g} seconds.'
              .format(k+1, trials, end-start))
    elif k is not None:
        print('Completed {:d} trials so far. This run took {:g} seconds.'
              .format(k+1, end-start))
    elif trials is not None:
        print('Completed one of {:d} trials. This run took {:g} seconds.'
              .format(trials, end-start))
    else:
        print('Completed one trial. This run took {:g} seconds.'
              .format(end-start))
    return result


def main(a, c, eps1, eps2, mle_trials, num_expt, filename):
    datasets = [multiple_experiments(a, c, eps1, eps2, num_expt)
                for _ in range(mle_trials)]

    model_fits = [mle_trial_wrapper(vels=dataset, two_par=True, N_obj=16,
                                    k=k, trials=mle_trials)
                  for (k, dataset) in enumerate(datasets)]
    model_fits = np.transpose(np.array(model_fits))
    model_fits = change_vars(model_fits, forward=False)

    save_array = np.reshape(model_fits, newshape=(4, -1), order='F').T

    trial_dir = 'dat-files/ml-trials/'
    np.savetxt(trial_dir + filename + '-trl.dat', save_array)


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    mle_trials = int(sys.argv[2])
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    pars = read_parameter_file(filename)
    main(mle_trials=mle_trials, **pars)
