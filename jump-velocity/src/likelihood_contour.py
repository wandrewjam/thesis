import numpy as np
import matplotlib.pyplot as plt
from jv import solve_pde, delta_h
from mle import change_vars


def full_objective(p, v, vmin, two_par, N_obj):
    """ Log likelihood function of the full model

    Parameters
    ----------
    N_obj
    p
    v
    vmin

    Returns
    -------
    float
    """
    # Transform the parameters back to a and epsilon
    if two_par:
        a, eps1 = change_vars(p, forward=False)[:, 0]
        c, eps2 = 1, np.inf
    else:
        a, eps1 = change_vars(p[:2], forward=False)[:, 0]
        c, eps2 = change_vars(p[2:], forward=False)[:, 0]

    h = 1. / N_obj
    s_eval = (np.arange(0, np.ceil(1. / (vmin * h))) + 1) * h

    # Define initial conditions for solve_pde
    y = np.linspace(0, 1, num=N_obj + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N_obj))

    u1_bdy = solve_pde(s_eval, p0, h, eps1=eps1, eps2=eps2, a=a,
                       b=1 - a, c=c, d=1-c, scheme='up')[3]
    pdf = np.interp(1. / v, s_eval,
                    u1_bdy / (1 - np.exp((a - 1) / eps1 + (c - 1) / eps2)))
    return -np.sum(np.log(pdf))


if __name__ == '__main__':
    import os
    # import sys

    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))
    sim_dir = 'dat-files/simulations/'
    vels = np.loadtxt(sim_dir + 'test-sim.dat')
    v_min = np.amin(vels)

    ap_plot = np.linspace(-1, 1, num=11)
    ep_plot = np.linspace(-3, 0, num=11)
    ap_eval = ap_plot[:-1] + (ap_plot[1] - ap_plot[0])/2
    ep_eval = ep_plot[:-1] + (ep_plot[1] - ep_plot[0])/2
    values = np.zeros(shape=ap_eval.shape * 2)

    for (j, ap) in enumerate(ap_eval):
        for (i, ep) in enumerate(ep_eval):
            values[i, j] = full_objective(np.array([ap, ep]), vels, vmin=v_min,
                                          two_par=True, N_obj=100)

    a_eval, e_eval = change_vars(np.stack([ap_eval, ep_eval], axis=0), forward=False)
    a_plot, e_plot = change_vars(np.stack([ap_plot, ep_plot], axis=0), forward=False)
    plt.contour(a_eval, e_eval, values, colors='k')
    plt.pcolormesh(a_plot, e_plot, values)
    plt.colorbar()
    plt.xlabel('$a$')
    plt.ylabel('$\\epsilon$')
    plt.show()
