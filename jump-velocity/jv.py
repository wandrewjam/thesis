import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, cumtrapz
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
    u, v, f, vf = np.split(p, 4)
    p0 = delta_h(-t, h)
    ka, kb = a / (1 + eps1 / eps2), b / (1 + eps1 / eps2)
    kc, kd = c / (1 + eps2 / eps1), d / (1 + eps2 / eps1)
    if scheme == 'up':
        du = (-1 / h * (-np.append(p0, u[:-1]) + u)
              + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u + ka * v + kc * f))
    elif scheme == 'bw':
        pm1 = delta_h(-h - t, h)
        du = (-1 / (2*h)
              * (np.append([pm1, p0], u[:-2]) - 4*np.append(p0, u[:-1]) + 3*u)
              + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u + ka * v + kc * f))
    else:
        raise ValueError('parameter \'scheme\' is not valid!')
    dv = (1 / eps1 + 1 / eps2) * (kb * u - (ka + kd) * v + kc * vf)
    df = (1 / eps1 + 1 / eps2) * (kd * u - (kb + kc) * f + ka * vf)
    dvf = (1 / eps1 + 1 / eps2) * (kd * v + kb * f - (ka + kc) * vf)
    return np.append(du, [dv, df, dvf])


def solve_pde(s_eval, p0, h, eps1, eps2, a, b, c, d, scheme):
    u0, v0, f0, vf0 = np.split(p0, 4)
    ka, kb = a / (1 + eps1 / eps2), b / (1 + eps1 / eps2)
    kc, kd = c / (1 + eps2 / eps1), d / (1 + eps2 / eps1)
    I = eye(u0.shape[0])
    dt = s_eval[1] - s_eval[0]
    rxn = bmat([[-(kb+kd)*I, ka*I, kc*I, None], [kb*I, -(ka+kd)*I, None, kc*I],
                [kd*I, None, -(kb+kc)*I, ka*I], [None, kd*I, kb*I, -(ka+kc)*I]])
    A = eye(4*u0.shape[0]) - (1 / eps1 + 1 / eps2) * dt*rxn
    solve = factorized(A)
    bc = delta_h(-s_eval, h)
    u, v, f, vf = u0[:, None], v0[:, None], f0[:, None], vf0[:, None]
    for i in range(s_eval.shape[0]-1):
        # rhsu = (u[:, -1] + dt/h*(np.append(bc[i], u[:-1, -1]) - u[:, -1])
        #         + (1 / eps1 + 1 / eps2) * dt/2*(-(kb + kd)*u[:, -1] + ka*v[:, -1] + kc*f[:, -1]))
        # rhsv = v[:, -1] + (1 / eps1 + 1 / eps2) * dt/2*(kb*u[:, -1] - (ka + kd)*v[:, -1] + kc*vf[:, -1])
        # rhsf = f[:, -1] + (1 / eps1 + 1 / eps2) * dt/2*(kd*u[:, -1] - (kc + kb)*f[:, -1] + ka*vf[:, -1])
        # rhsvf = vf[:, -1] + (1 / eps1 + 1 / eps2) * dt/2*(kd*v[:, -1] + kb*f[:, -1] - (ka + kc)*vf[:, -1])
        rhsu = u[:, -1] + dt/h*(np.append(bc[i], u[:-1, -1]) - u[:, -1])
        rhsv = v[:, -1]
        rhsf = f[:, -1]
        rhsvf = vf[:, -1]
        rhs = np.hstack([rhsu, rhsv, rhsf, rhsvf])
        p = solve(rhs)
        utemp, vtemp, ftemp, vftemp = np.split(p, 4)
        u = np.append(u, utemp[:, None], axis=1)
        v = np.append(v, vtemp[:, None], axis=1)
        f = np.append(f, ftemp[:, None], axis=1)
        vf = np.append(vf, vftemp[:, None], axis=1)
    return u, v, f, vf


def delta_h(x, h):
    return phi(x/h) / h
    # return 1 - np.cos(2*np.pi*x)


def phi(r):
    condlist = [np.abs(r) <= 2]
    choicelist = [(1 + np.cos(np.pi * r / 2)) / 4]
    # choicelist = [315*(r-2)**4*(r+2)**4/131072]
    return np.select(condlist, choicelist)


def slow_v(y, t):
    return (1 / np.sqrt(4 * np.pi * eps1 * a * b * t)
            * np.exp(-(y - a*t)**2 / (4 * eps1 * a * b * t)))


def fast_w(y, t):
    return (-(y - a*t) / (4 * t * np.sqrt(np.pi * eps1 * a * b * t))
            * np.exp(-(y - a*t)**2 / (4 * eps1 * a * b * t)))


def read_parameter_file(filename):
    parlist = [('filename', filename.split('.')[0])]

    with open(filename) as f:
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


def main(N, eps1, eps2, a, c, s_max, s_samp, scheme, filename, show_plots=False):
    def animate(i):
        line_u.set_ydata(np.append(delta_h(-s_eval[i], h), u_data[:, i]))
        line_v.set_ydata(v_data[:, i])
        line_f.set_ydata(f_data[:, i])
        line_vf.set_ydata(vf_data[:, i])
        vline.set_xdata([s_eval[i]] * 2)
        return line_u, line_v, line_f, line_vf, vline

    b, d = 1 - a, 1 - c
    h = 1. / N
    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(3 * N))

    # Enforce the CFL condition
    if scheme == 'up':
        max_step = h
        s_eval = np.arange(start=0, stop=s_max + max_step, step=max_step)
        u_data, v_data, f_data, vf_data = solve_pde(s_eval, p0, h=h, eps1=eps1, eps2=eps2, a=a, b=b, c=c, d=d, scheme=scheme)
    elif scheme == 'bw':
        max_step = 2 * h
        s_eval = np.arange(start=0, stop=s_max + max_step, step=max_step)
        u_data, v_data, f_data, vf_data = solve_pde(s_eval, p0, h=h, eps1=eps1, eps2=eps2, a=a, b=b, c=c, d=d, scheme=scheme)
    elif scheme == 'MoL':
        max_step = h
        s_eval = np.linspace(0, s_max, s_samp)
        sol = odeint(four_state, y0=p0, t=s_eval, args=(h, eps1, eps2, a, b, c, d), hmax=max_step)
        u_data, v_data, f_data, vf_data = np.split(sol.T, 4)
    else:
        raise ValueError('parameter \'scheme\' is not valid!')

    fig, ax = plt.subplots()
    line_u, line_v, line_f, line_vf = ax.plot(
        y, np.append(delta_h(s_eval[0], h), u_data[:, 0]), y[1:], v_data[:, 0],
        y[1:], f_data[:, 0], y[1:], vf_data[:, 0])

    line_u.set_label('$q_U$')
    line_v.set_label('$q_V$')
    line_f.set_label('$q_F$')
    line_vf.set_label('$q_{VF}$')

    ax.set_ylim(bottom=-1, top=11)
    vline = ax.axvline(s_eval[0], color='k')
    vline.set_label('$y = t$')

    ax.legend(loc='upper right')
    ax.set_xlabel('$y$')
    ax.set_ylabel('Probability density')

    ani = animation.FuncAnimation(
        fig, animate, frames=s_eval.shape[0], interval=25)
    ani.save('ani_' + filename + '.mp4')
    if show_plots:
        plt.show()

    s_mask = s_eval > 2./3
    fig_av, ax_av = plt.subplots()
    ax_av.plot(1 / s_eval[s_mask], s_eval[s_mask] ** 2 * u_data[-1, s_mask], 1 / s_eval[s_mask], 1 - cumtrapz(u_data[-1, 1:], 1 / s_eval[1:], initial=0)[s_mask[1:]])
    # ax_av.plot(1 / s_eval[s_mask], 1 - cumtrapz(u_data[-1, :], 1 / s_eval, initial=0)[s_mask])
    ax_av.axvline(1, color='k')
    ax_av.set_xlabel('$v^*$')
    ax_av.set_ylabel('Probability density')
    fig_av.savefig('avg-vel_' + filename + '.png')
    if show_plots:
        plt.show()

    plt.plot(s_eval, u_data[-1, :], s_eval, cumtrapz(u_data[-1, :], s_eval, initial=0))
    plt.show()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]

    pars = read_parameter_file(filename)
    main(show_plots=True, **pars)

    # Adiabatic reduction

