import numpy as np
import matplotlib.pyplot as plt
from jv import solve_pde, delta_h
from mle import change_vars

if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    ap_vec, ep_vec = np.linspace(-1, 1, num=6), np.linspace(-3, -1, num=6)

    a_vec, e_vec = change_vars(np.stack([ap_vec, ep_vec], axis=0),
                               forward=False)
    N_obj = 64
    h = 1. / N_obj
    vmin = .1
    s_eval = (np.arange(0, np.ceil(1. / (vmin * h))) + 1) * h
    s_mask = s_eval > 1.

    y = np.linspace(0, 1, num=N_obj + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N_obj))

    pdf_array = np.zeros(shape=s_eval[s_mask].shape + ap_vec.shape
                         + ep_vec.shape)

    for (j, a) in enumerate(a_vec):
        for (k, eps1) in enumerate(e_vec):
            u1_bdy = solve_pde(s_eval, p0, h, eps1=eps1, eps2=np.inf, a=a,
                               b=1-a, c=.5, d=.5, scheme='up')[3]
            pdf_array[:, j, k] = (s_eval[s_mask] ** 2 * u1_bdy[s_mask])\
                                / (1 - np.exp((a - 1) / eps1))

    # Plot stuff
    # slices in v
    for k in range(0, e_vec.size, (e_vec.size-1) // 2):
        for j in range(a_vec.size):
            plt.plot(1./s_eval[s_mask], pdf_array[:, j, k])
        plt.title('$\\epsilon = {:g}$'.format(e_vec[k]))
        plt.xlabel('$v^*$')
        plt.ylabel('Probability density')
        plt.show()

    for j in range(0, a_vec.size, (a_vec.size-1) // 2):
        for k in range(e_vec.size):
            plt.plot(1./s_eval[s_mask], pdf_array[:, j, k])
        plt.title('$a = {:g}$'.format(a_vec[j]))
        plt.xlabel('$v^*$')
        plt.ylabel('Probability density')
        plt.show()
