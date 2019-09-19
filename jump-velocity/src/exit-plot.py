import numpy as np
import matplotlib.pyplot as plt
from simulate import experiment, read_parameter_file
from jv import delta_h, solve_pde


def main(a, c, eps1, eps2, num_expt, filename):
    sim_dir = 'dat-files/simulations/'
    exits = np.loadtxt(sim_dir + filename + '-exit.dat')

    N = 1024
    s_max = 50
    s_eval = np.linspace(0, s_max, num=s_max * N + 1)
    h = 1. / N
    scheme = None
    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N))

    pde_sol = solve_pde(s_eval, p0, h, eps1, eps2, a, 1 - a, c, 1 - c, scheme)
    y_store = pde_sol[0]
    t_store = pde_sol[1]
    pde_exit = pde_sol[-1]
    marg_y = np.trapz(pde_exit, s_eval[t_store], axis=1)
    marg_s = np.trapz(pde_exit, y[y_store], axis=0)

    plt.hist(exits[:, 0], density=True)
    plt.plot(y[y_store], marg_y)
    plt.show()

    plt.step(np.sort(exits[:, 1]), np.linspace(0, 1, num=exits.shape[0]))
    plt.plot(s_eval[t_store], marg_s)
    plt.show()
    return None


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    pars = read_parameter_file(filename)
    main(**pars)
