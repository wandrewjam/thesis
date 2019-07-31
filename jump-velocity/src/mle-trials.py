import numpy as np
from simulate import read_parameter_file, multiple_experiments
from mle import fit_models


def main(a, c, eps1, eps2, mle_trials, trial_size):
    datasets = [multiple_experiments(a, c, eps1, eps2, trial_size)
                for _ in range(mle_trials)]

    model_fits = [fit_models(dataset, two_par=True) for dataset in datasets]

    # This should generate the fits, now I just have to save the data
    return None


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    mle_trials = int(sys.argv[2])
    trial_size = int(sys.argv[3])
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    pars = read_parameter_file(filename)
    main(mle_trials=mle_trials, trial_size=trial_size, **pars)
