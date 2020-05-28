import numpy as np
from sphere_integration_utils import generate_grid, compute_helper_funs
from timeit import default_timer as timer


if __name__ == '__main__':
    n_nodes = 24
    eps = 0.01

    # Generate mesh and r array
    nodes = generate_grid(n_nodes, a=1.5, b=0.5)[2]
    x0 = nodes
    xe = nodes
    del_x0 = xe[:, np.newaxis, :] - x0[np.newaxis, :, :]
    r0 = np.linalg.norm(del_x0, axis=-1)[:, :, np.newaxis, np.newaxis]
    x_im = np.array([-1, 1, 1]) * x0
    del_x = xe[:, np.newaxis, :] - x_im[np.newaxis, :, :]
    r = np.linalg.norm(del_x, axis=-1)[:, :, np.newaxis, np.newaxis]

    # Pre-compute square root
    r_pre = np.linspace(0, np.amax(r), num=10**4)
    sq = np.sqrt(r_pre**2 + eps**2)

    # Pre-compute H and D functions
    h1_pre, h2_pre, d1_pre, d2_pre, h1p_pre, h2p_pre = \
        compute_helper_funs(r_pre, eps=eps)

    # Look-up square roots
    start = timer()
    sre = np.interp(r, r_pre, sq)
    h1 = 1 / (8 * np.pi * sre) + eps**2 / (8 * np.pi * sre ** 3)
    h2 = 1 / (8 * np.pi * sre ** 3)
    d1 = (4 * np.pi * sre ** 3) ** (-1) - 3 * eps**2 / (4 * np.pi * sre ** 5)
    d2 = - 3 / (4 * np.pi * sre ** 5)
    h1p = -r / (8 * np.pi * sre**3) - 3 * r * eps**2 / (8 * np.pi * sre**5)
    h2p = -3 * r / (8 * np.pi * sre**5)
    end = timer()

    print('Pre-compute square root time: {}'.format(end-start))

    # Look-up H and D values w/o sorting
    start = timer()
    h1 = np.interp(r, r_pre, h1_pre)
    h2 = np.interp(r, r_pre, h2_pre)
    d1 = np.interp(r, r_pre, d1_pre)
    d2 = np.interp(r, r_pre, d2_pre)
    h1p = np.interp(r, r_pre, h1p_pre)
    h2p = np.interp(r, r_pre, h2p_pre)
    end = timer()
    print('Pre-compute H and D time: {}'.format(end-start))

    # Look-up H and D values w/ sorting (Someday)

    # Compute H and D on the fly
    start = timer()
    h1, h2, d1, d2, h1p, h2p = compute_helper_funs(r, eps=eps)
    end = timer()

    print('Compute H and D regular: {}'.format(end-start))
