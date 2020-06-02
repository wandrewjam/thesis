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
    ind_u = np.triu_indices_from(r[..., 0, 0])

    # Pre-compute square root
    r_pre = np.linspace(0, np.amax(r), num=10**4)
    del_r = r_pre[1] - r_pre[0]
    sq = np.sqrt(r_pre**2 + eps**2)

    # Pre-compute H and D functions
    h1_pre, h2_pre, d1_pre, d2_pre, h1p_pre, h2p_pre = \
        compute_helper_funs(r_pre, eps=eps)

    # Interpolate square roots
    start = timer()
    sre = np.interp(r, r_pre, sq)
    h1 = 1 / (8 * np.pi * sre) + eps**2 / (8 * np.pi * sre ** 3)
    h2 = 1 / (8 * np.pi * sre ** 3)
    d1 = (4 * np.pi * sre ** 3) ** (-1) - 3 * eps**2 / (4 * np.pi * sre ** 5)
    d2 = - 3 / (4 * np.pi * sre ** 5)
    h1p = -r / (8 * np.pi * sre**3) - 3 * r * eps**2 / (8 * np.pi * sre**5)
    h2p = -3 * r / (8 * np.pi * sre**5)
    end = timer()

    print('Interpolate square root time: {}'.format(end-start))

    # Look-up square roots
    start = timer()
    look_ups = np.round(r / del_r, 0).astype(dtype='int')
    sre = sq[look_ups]
    h1 = 1 / (8 * np.pi * sre) + eps**2 / (8 * np.pi * sre ** 3)
    h2 = 1 / (8 * np.pi * sre ** 3)
    d1 = (4 * np.pi * sre ** 3) ** (-1) - 3 * eps**2 / (4 * np.pi * sre ** 5)
    d2 = - 3 / (4 * np.pi * sre ** 5)
    h1p = -r / (8 * np.pi * sre**3) - 3 * r * eps**2 / (8 * np.pi * sre**5)
    h2p = -3 * r / (8 * np.pi * sre**5)
    end = timer()

    print('Look-up square root time: {}'.format(end-start))

    # Interpolate square roots (symmetric)
    start = timer()
    r_sym = r[..., 0, 0][ind_u]
    h1_arr, h2_arr, d1_arr, d2_arr, h1p_arr, h2p_arr = [
        np.zeros(shape=r.shape[:2]) for _ in range(6)]
    sre_sym = np.interp(r_sym, r_pre, sq)
    h1 = 1 / (8 * np.pi * sre_sym) + eps ** 2 / (8 * np.pi * sre_sym ** 3)
    h2 = 1 / (8 * np.pi * sre_sym ** 3)
    d1 = (4 * np.pi * sre_sym ** 3) ** (-1) - 3 * eps ** 2 / (4 * np.pi * sre_sym ** 5)
    d2 = - 3 / (4 * np.pi * sre_sym ** 5)
    h1p = -r_sym / (8 * np.pi * sre_sym ** 3) - 3 * r_sym * eps ** 2 / (8 * np.pi * sre_sym ** 5)
    h2p = -3 * r_sym / (8 * np.pi * sre_sym ** 5)

    h1_arr[ind_u], h1_arr[ind_u[::-1]] = h1, h1
    h2_arr[ind_u], h2_arr[ind_u[::-1]] = h2, h2
    d1_arr[ind_u], d1_arr[ind_u[::-1]] = d1, d1
    d2_arr[ind_u], d2_arr[ind_u[::-1]] = d2, d2
    h1p_arr[ind_u], h1p_arr[ind_u[::-1]] = h1p, h1p
    h2p_arr[ind_u], h2p_arr[ind_u[::-1]] = h2p, h2p
    end = timer()

    print('Interpolate symmetric square root time: {}'.format(end-start))

    # Interpolate H and D values w/o sorting
    start = timer()
    r_sym = r[..., 0, 0][ind_u]
    h1_arr, h2_arr, d1_arr, d2_arr, h1p_arr, h2p_arr = [
        np.zeros(shape=r.shape[:2]) for _ in range(6)]
    h1 = np.interp(r_sym, r_pre, h1_pre)
    h2 = np.interp(r_sym, r_pre, h2_pre)
    d1 = np.interp(r_sym, r_pre, d1_pre)
    d2 = np.interp(r_sym, r_pre, d2_pre)
    h1p = np.interp(r_sym, r_pre, h1p_pre)
    h2p = np.interp(r_sym, r_pre, h2p_pre)

    h1_arr[ind_u], h1_arr[ind_u[::-1]] = h1, h1
    h2_arr[ind_u], h2_arr[ind_u[::-1]] = h2, h2
    d1_arr[ind_u], d1_arr[ind_u[::-1]] = d1, d1
    d2_arr[ind_u], d2_arr[ind_u[::-1]] = d2, d2
    h1p_arr[ind_u], h1p_arr[ind_u[::-1]] = h1p, h1p
    h2p_arr[ind_u], h2p_arr[ind_u[::-1]] = h2p, h2p
    end = timer()
    print('Interpolate symmetric H and D time: {}'.format(end-start))

    # Interpolate H and D values (symmetric) w/o sorting
    start = timer()
    h1 = np.interp(r, r_pre, h1_pre)
    h2 = np.interp(r, r_pre, h2_pre)
    d1 = np.interp(r, r_pre, d1_pre)
    d2 = np.interp(r, r_pre, d2_pre)
    h1p = np.interp(r, r_pre, h1p_pre)
    h2p = np.interp(r, r_pre, h2p_pre)
    end = timer()
    print('Interpolate H and D time: {}'.format(end-start))

    # Look-up H and D values
    start = timer()
    look_ups = np.round(r / del_r, 0).astype(dtype='int')
    h1 = h1_pre[look_ups]
    h2 = h2_pre[look_ups]
    d1 = d1_pre[look_ups]
    d2 = d2_pre[look_ups]
    h1p = h1p_pre[look_ups]
    h2p = h2p_pre[look_ups]
    end = timer()
    print('Look-up H and D time: {}'.format(end-start))

    # Compute H and D on the fly
    start = timer()
    h1, h2, d1, d2, h1p, h2p = compute_helper_funs(r, eps=eps)
    end = timer()

    print('Compute H and D regular: {}'.format(end-start))
