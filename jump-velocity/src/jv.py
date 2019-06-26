import numpy as np
from scipy.integrate import odeint
from scipy.sparse import eye, bmat
from scipy.sparse.linalg import factorized


def two_state(t, p, h, eps1, a, b, scheme='up'):
    u, v = np.split(p, 2)
    p0 = delta_h(-t, h)
    if scheme == 'up':
        du = (-1 / h * (-np.append(p0, u[:-1]) + u)
              + 1 / eps1 * (-b * u + a * v))
    elif scheme == 'bw':
        pm1 = delta_h(-h - t, h)
        du = (-1 / (2*h)
              * (np.append([pm1, p0], u[:-2]) - 4*np.append(p0, u[:-1]) + 3*u)
              + 1 / eps1 * (-b * u + a * v))
    else:
        raise ValueError('parameter \'scheme\' is not valid!')
    dv = 1 / eps1 * (b * u - a * v)
    return np.append(du, dv)


def four_state(p, t, h, eps1, eps2, a, b, c, d, scheme='up'):
    u0, u1, v, f, vf = np.split(p, 5)
    p0 = delta_h(-t, h)
    ka, kb = a / (1 + eps1 / eps2), b / (1 + eps1 / eps2)
    kc, kd = c / (1 + eps2 / eps1), d / (1 + eps2 / eps1)
    if scheme == 'up':
        du0 = (-1 / h * (-np.append(p0, u0[:-1]) + u0)
               + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u0))
        du1 = (-1 / h * (-np.append(0, u1[:-1]) + u1)
               + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u1 + ka * v + kc * f))
    elif scheme == 'bw':
        pm1 = delta_h(-h - t, h)
        du0 = (-1 / (2*h)
               * (np.append([pm1, p0], u0[:-2]) - 4*np.append(p0, u0[:-1]) + 3*u0)
               + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u0))
        du1 = (-1 / (2*h)
               * (np.append([0, 0], u1[:-2]) - 4*np.append(p0, u1[:-1]) + 3*u1)
               + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u1 + ka * v + kc * f))
    else:
        raise ValueError('parameter \'scheme\' is not valid!')
    dv = (1 / eps1 + 1 / eps2) * (kb * (u0 + u1) - (ka + kd) * v + kc * vf)
    df = (1 / eps1 + 1 / eps2) * (kd * (u0 + u1) - (kb + kc) * f + ka * vf)
    dvf = (1 / eps1 + 1 / eps2) * (kd * v + kb * f - (ka + kc) * vf)
    return np.append(du0, [du1, dv, df, dvf])


def solve_pde(s_eval, p0, h, eps1, eps2, a, b, c, d, scheme, fast=True, s_samp=1000):
    u00, u10, v0, f0, vf0 = np.split(p0, 5)
    N = u00.shape[0]
    ystore = np.linspace(0, N, num=101, dtype='int')[1:] - 1
    tstore = np.linspace(0, s_eval.shape[0]-1, num=s_samp+1, dtype='int')
    ka, kb = a / (1 + eps1 / eps2), b / (1 + eps1 / eps2)
    kc, kd = c / (1 + eps2 / eps1), d / (1 + eps2 / eps1)
    dt = s_eval[1] - s_eval[0]
    bc = delta_h(-s_eval, h)
    u0, u1, v, f, vf = u00[ystore, None], u10[ystore, None], v0[ystore, None], f0[ystore, None], vf0[ystore, None]
    u0_bdy, u1_bdy = u00[-1], u10[-1]
    if fast:
        rxn = np.array([[-(kb+kd), 0, 0, 0, 0], [0, -(kb+kd), ka, kc, 0], [kb, kb, -(ka+kd), 0, kc],
                        [kd, kd, 0, -(kb+kc), ka], [0, 0, kd, kb, -(ka+kc)]])
        A = np.eye(5) - (1 / eps1 + 1 / eps2) * dt * rxn
        Ai = np.linalg.inv(A)
        u0temp, u1temp, vtemp, ftemp, vftemp = u00, u10, v0, f0, vf0
        for i in range(1, s_eval.shape[0]):
            rhsu0 = u0temp + dt/h*(np.append(bc[i-1], u0temp[:-1]) - u0temp)
            rhsu1 = u1temp + dt/h*(np.append(0, u1temp[:-1]) - u1temp)
            rhsv = vtemp
            rhsf = ftemp
            rhsvf = vftemp
            u0temp = Ai[0, 0] * rhsu0 + Ai[0, 1] * rhsu1 + Ai[0, 2] * rhsv + Ai[0, 3] * rhsf + Ai[0, 4] * rhsvf
            u1temp = Ai[1, 0] * rhsu0 + Ai[1, 1] * rhsu1 + Ai[1, 2] * rhsv + Ai[1, 3] * rhsf + Ai[1, 4] * rhsvf
            vtemp = Ai[2, 0] * rhsu0 + Ai[2, 1] * rhsu1 + Ai[2, 2] * rhsv + Ai[2, 3] * rhsf + Ai[2, 4] * rhsvf
            ftemp = Ai[3, 0] * rhsu0 + Ai[3, 1] * rhsu1 + Ai[3, 2] * rhsv + Ai[3, 3] * rhsf + Ai[3, 4] * rhsvf
            vftemp = Ai[4, 0] * rhsu0 + Ai[4, 1] * rhsu1 + Ai[4, 2] * rhsv + Ai[4, 3] * rhsf + Ai[4, 4] * rhsvf
            u0_bdy = np.append(u0_bdy, u0temp[-1])
            u1_bdy = np.append(u1_bdy, u1temp[-1])
            if i in tstore:
                u0 = np.append(u0, u0temp[ystore, None], axis=1)
                u1 = np.append(u1, u1temp[ystore, None], axis=1)
                v = np.append(v, vtemp[ystore, None], axis=1)
                f = np.append(f, ftemp[ystore, None], axis=1)
                vf = np.append(vf, vftemp[ystore, None], axis=1)
    else:
        I = eye(N, format='csc')
        rxn = bmat([[-(kb+kd)*I, None, None, None, None], [None, -(kb+kd)*I, ka*I, kc*I, None], [kb*I, kb*I, -(ka+kd)*I, None, kc*I],
                    [kd*I, kd*I, None, -(kb+kc)*I, ka*I], [None, None, kd*I, kb*I, -(ka+kc)*I]], format='csc')
        A = eye(5*N, format='csc') - (1 / eps1 + 1 / eps2) * dt * rxn
        solve = factorized(A)
        for i in range(s_eval.shape[0]-1):
            rhsu0 = u0[:, -1] + dt/h*(np.append(bc[i], u0[:-1, -1]) - u0[:, -1])
            rhsu1 = u1[:, -1] + dt/h*(np.append(0, u1[:-1, -1]) - u1[:, -1])
            rhsv = v[:, -1]
            rhsf = f[:, -1]
            rhsvf = vf[:, -1]
            rhs = np.hstack([rhsu0, rhsu1, rhsv, rhsf, rhsvf])
            p = solve(rhs)
            u0temp, u1temp, vtemp, ftemp, vftemp = np.split(p, 5)
            u0 = np.append(u0, u0temp[:, None], axis=1)
            u1 = np.append(u1, u1temp[:, None], axis=1)
            v = np.append(v, vtemp[:, None], axis=1)
            f = np.append(f, ftemp[:, None], axis=1)
            vf = np.append(vf, vftemp[:, None], axis=1)
    return ystore, tstore, u0_bdy, u1_bdy, u0, u1, v, f, vf


def delta_h(x, h):
    return phi(x/h) / h


def phi(r):
    condlist = [np.abs(r) <= 2]
    choicelist = [(1 + np.cos(np.pi * r / 2)) / 4]
    return np.select(condlist, choicelist)


def slow_v(y, t):
    return (1 / np.sqrt(4 * np.pi * eps1 * a * b * t)
            * np.exp(-(y - a*t)**2 / (4 * eps1 * a * b * t)))


def fast_w(y, t):
    return (-(y - a*t) / (4 * t * np.sqrt(np.pi * eps1 * a * b * t))
            * np.exp(-(y - a*t)**2 / (4 * eps1 * a * b * t)))


def read_parameter_file(filename):
    txt_dir = '../par-files/'
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
            elif key == 'scheme':
                parlist.append((key, value))
            else:
                parlist.append((key, float(value)))
    return dict(parlist)


def main(N, eps1, eps2, a, c, s_max, s_samp, scheme, filename):
    b, d = 1 - a, 1 - c
    h = 1. / N
    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N))

    # Enforce the CFL condition
    if scheme == 'up':
        max_step = h
        s_eval = np.linspace(start=0, stop=s_max, num=np.ceil(s_max / max_step)+1)
        y_store, t_store, u0_bdy, u1_bdy, u0_data, u1_data, v_data, f_data, vf_data = (
            solve_pde(s_eval, p0, h=h, eps1=eps1, eps2=eps2,
                      a=a, b=b, c=c, d=d, scheme=scheme, s_samp=s_samp))
    elif scheme == 'bw':
        max_step = 2 * h
        s_eval = np.arange(start=0, stop=s_max + max_step, step=max_step)
        u0_data, u1_data, v_data, f_data, vf_data = solve_pde(s_eval, p0, h=h, eps1=eps1, eps2=eps2, a=a, b=b, c=c, d=d, scheme=scheme)
    elif scheme == 'MoL':
        max_step = h
        s_eval = np.linspace(0, s_max, s_samp + 1)
        sol = odeint(four_state, y0=p0, t=s_eval, args=(h, eps1, eps2, a, b, c, d), hmax=max_step)
        u0_data, u1_data, v_data, f_data, vf_data = np.split(sol.T, 5)
        u0_bdy, u1_bdy = u0_data[-1, :], u1_data[-1, :]
        y_store = np.linspace(0, N, num=101, dtype='int')[1:] - 1
        t_store = np.linspace(0, s_eval.shape[0]-1, num=s_samp+1, dtype='int')
    else:
        raise ValueError('parameter \'scheme\' is not valid!')

    npz_dir = '../npz-files/'
    np.savez_compressed(npz_dir + filename, y_store, t_store, y, s_eval, u0_data, u1_data, v_data, f_data, vf_data, u0_bdy, u1_bdy,
                        y_store=y_store, t_store=t_store, y=y, s_eval=s_eval, u0_data=u0_data, u1_data=u1_data, v_data=v_data,
                        f_data=f_data, vf_data=vf_data, u0_bdy=u0_bdy, u1_bdy=u1_bdy)


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]

    pars = read_parameter_file(filename)
    main(**pars)
