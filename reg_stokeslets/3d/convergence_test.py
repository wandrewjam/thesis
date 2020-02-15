import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sphere_integration_utils import l2_error, sphere_integrate


def main(proc=1, plot_result=False):
    # Test convergence of Regularized Stokeslets
    errs = list()
    n_nodes = 12 + 36 * np.arange(6)

    for n in n_nodes:
        start = timer()

        def error(x):
            return l2_error(x, n, proc)

        errs.append(sphere_integrate(error, n_nodes=n))
        end = timer()

        print('Total time for {} nodes: {} seconds'.format(n, end - start))

    grid_size = np.pi / (2 * n_nodes)
    save_array = np.array([grid_size, errs, n_nodes]).T

    np.savetxt('convergence_test.dat', save_array)

    if plot_result:
        plt.plot(grid_size, errs)
        plt.show()


if __name__ == '__main__':
    main()
