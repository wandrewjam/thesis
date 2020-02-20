import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sphere_integration_utils import l2_error, sphere_integrate


def main(filename, proc=1, plot_result=False):
    # Test to see if a refinement file already exists
    import os.path
    filename = filename + '.dat'
    if os.path.exists(filename):
        raise ValueError('{} already exists!'.format(filename))

    # Test convergence in the blob parameter
    epsilon = np.linspace(0.01, 0.15, num=21)
    n_nodes = 24

    errs = list()
    for eps in epsilon:
        start = timer()

        def error(x):
            return l2_error(x, n_nodes, proc, eps=eps)

        errs.append(sphere_integrate(error, n_nodes=n_nodes))
        end = timer()

        print('Total time for epsilon = {}: {} seconds'
              .format(eps, end - start))

    save_array = np.array([errs, epsilon]).T

    np.savetxt(filename, save_array)

    if plot_result:
        plt.plot(epsilon, errs)
        plt.show()


if __name__ == '__main__':
    main('epsilon_refinement', proc=4)
