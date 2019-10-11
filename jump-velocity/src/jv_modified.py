import numpy as np

from src.jv import delta_h


def read_parameter_file(filename):
    txt_dir = 'par-files/'
    parlist = [('filename', filename)]

    with open(txt_dir + filename + '.txt') as f:
        while True:
            command = f.readline().split()
            if len(command) < 1:
                continue
            if command[0] == 'done':
                break

            key, value = command
            if key == 'N' or key == 's_samp':
                parlist.append((key, int(value)))
            else:
                parlist.append((key, float(value)))
    return dict(parlist)


def solve_mod_pde(s_eval, p0, h, alpha, beta, gamma, delta, eta, lam, s_samp=1024):
    u0, b0, bf0, T0 = np.split(p0, 4)
    N = u0.shape[0]
    yi = np.linspace(0, 1, num=N+1)
    ystore = np.linspace(0, N, num=129, dtype='int')[1:] - 1
    tstore = np.linspace(0, s_eval.shape[0]-1, num=s_samp+1, dtype='int')
    dt = s_eval[1] - s_eval[0]
    bc = delta_h(-s_eval, h)
    u, b, bf, T = u0[ystore, None], b0[ystore, None], bf0[ystore, None], T0[ystore, None]

    rxn = np.array([[-beta, alpha, 0, 0], [beta, -(alpha+delta), gamma, lam],
                    [0, delta, -(gamma + eta), 0], [0, 0, eta, -lam]])
    A = np.eye(4) - dt * rxn
    Ai = np.linalg.inv(A)
    utemp, btemp, bftemp, Ttemp = u0, b0, bf0, T0
    u_bdy, T_bdy = u0[-1], T0[-1]
    for i in range(1, s_eval.shape[0]):
        rhsu = utemp + dt/h*(np.append(bc[i-1], utemp[:-1]) - utemp)
        rhsb = btemp
        rhsbf = bftemp
        rhsT = Ttemp + dt/h*(np.append(0, Ttemp[:-1]) - Ttemp)

        utemp = Ai[0, 0] * rhsu + Ai[0, 1] * rhsb + Ai[0, 2] * rhsbf + Ai[0, 3] * rhsT
        btemp = Ai[1, 0] * rhsu + Ai[1, 1] * rhsb + Ai[1, 2] * rhsbf + Ai[1, 3] * rhsT
        bftemp = Ai[2, 0] * rhsu + Ai[2, 1] * rhsb + Ai[2, 2] * rhsbf + Ai[2, 3] * rhsT
        Ttemp = Ai[3, 0] * rhsu + Ai[3, 1] * rhsb + Ai[3, 2] * rhsbf + Ai[3, 3] * rhsT

        u_bdy = np.append(u_bdy, utemp[-1])
        T_bdy = np.append(T_bdy, Ttemp[-1])

        if i in tstore:
            u = np.append(u, utemp[ystore, None], axis=1)
            b = np.append(b, btemp[ystore, None], axis=1)
            bf = np.append(bf, bftemp[ystore, None], axis=1)
            T = np.append(T, Ttemp[ystore, None], axis=1)

    return ystore, tstore, u_bdy, T_bdy, u, b, bf, T


def main(filename, N, alpha, beta, gamma, delta, eta, lam, s_max, s_samp):
    h = 1. / N
    y = np.linspace(0, 1, num=N+1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(3 * N))

    max_step = h
    s_eval = np.linspace(start=0, stop=s_max, num=np.ceil(s_max / max_step)+1)
    y_store, t_store, u_bdy, T_bdy, u_data, b_data, bf_data, T_data = (
        solve_mod_pde(s_eval, p0, h, alpha, beta, gamma, delta, eta,
                      lam, s_max, s_samp)
    )

    npz_dir = 'npz-files/'
    np.savez_compressed(
        npz_dir + filename, y_store, t_store, y, s_eval, u_data, b_data,
        bf_data, T_data, u_bdy, T_bdy, y_store=y_store, t_store=t_store, y=y,
        s_eval=s_eval, u_data=u_data, b_data=b_data, bf_data=bf_data,
        T_data=T_data, u_bdy=u_bdy, T_bdy=T_bdy
    )
    return None


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    pars = read_parameter_file(filename)
    main(**pars)
