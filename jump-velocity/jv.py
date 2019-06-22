import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp, cumtrapz


def two_state(t, p, scheme='up'):
    u, v = np.split(p, 2)
    p0 = delta_h(-t)
    if scheme == 'up':
        du = (-1 / h * (-np.append(p0, u[:-1]) + u)
              + 1 / eps1 * (-b * u + a * v))
    elif scheme == 'bw':
        pm1 = delta_h(-h-t)
        du = (-1 / (2*h)
              * (np.append([pm1, p0], u[:-2]) - 4*np.append(p0, u[:-1]) + 3*u)
              + 1 / eps1 * (-b * u + a * v))
    else:
        raise ValueError('parameter \'scheme\' is not valid!')
    dv = 1 / eps1 * (b * u - a * v)
    return np.append(du, dv)


def four_state(t, p, scheme='up'):
    u, v, f, vf = np.split(p, 4)
    p0 = delta_h(-t)
    ka, kb = a / (1 + eps1 / eps2), b / (1 + eps1 / eps2)
    kc, kd = c / (1 + eps2 / eps1), d / (1 + eps2 / eps1)
    if scheme == 'up':
        du = (-1 / h * (-np.append(p0, u[:-1]) + u)
              + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u + ka * v + kc * f))
    elif scheme == 'bw':
        pm1 = delta_h(-h-t)
        du = (-1 / (2*h)
              * (np.append([pm1, p0], u[:-2]) - 4*np.append(p0, u[:-1]) + 3*u)
              + (1 / eps1 + 1 / eps2) * (-(kb + kd) * u + ka * v + kc * f))
    else:
        raise ValueError('parameter \'scheme\' is not valid!')
    dv = (1 / eps1 + 1 / eps2) * (kb * u - (ka + kd) * v + kc * vf)
    df = (1 / eps1 + 1 / eps2) * (kd * u - (kb + kc) * f + ka * vf)
    dvf = (1 / eps1 + 1 / eps2) * (kd * v + kb * f - (ka + kc) * vf)
    return np.append(du, [dv, df, dvf])


def delta_h(x):
    return phi(x/h) / h
    # return 1 - np.cos(2*np.pi*x)


def phi(r):
    condlist = [np.abs(r) <= 2]
    choicelist = [(1 + np.cos(np.pi * r / 2)) / 4]
    # choicelist = [315*(r-2)**4*(r+2)**4/131072]
    return np.select(condlist, choicelist)


def init():
    line_u.set_ydata([np.nan] * y.shape[0])
    line_v.set_ydata([np.nan] * y.shape[0])
    line_f.set_ydata([np.nan] * y.shape[0])
    line_vf.set_ydata([np.nan] * y.shape[0])
    vline.set_xdata([np.nan] * 2)
    return line_u, line_v, line_f, line_vf, vline


def animate(i):
    line_u.set_ydata(np.append(delta_h(-sol.t[i]), u_data[:, i]))
    line_v.set_ydata(v_data[:, i])
    line_f.set_ydata(f_data[:, i])
    line_vf.set_ydata(vf_data[:, i])
    vline.set_xdata([sol.t[i]] * 2)
    return line_u, line_v, line_f, line_vf, vline


def slow_v(y, t):
    return (1 / np.sqrt(4 * np.pi * eps1 * a * b * t)
            * np.exp(-(y - a*t)**2 / (4 * eps1 * a * b * t)))


def fast_w(y, t):
    return (-(y - a*t) / (4 * t * np.sqrt(np.pi * eps1 * a * b * t))
            * np.exp(-(y - a*t)**2 / (4 * eps1 * a * b * t)))


if __name__ == '__main__':
    N = 50
    eps1, eps2 = .1, np.inf
    a, c = .5, .5
    b, d = 1 - a, 1 - c
    s_max = 10
    scheme = 'up'

    h = 1./N
    y = np.linspace(0, 1, num=N + 1)

    u_init = delta_h(y[1:])
    p0 = np.append(u_init, np.zeros(3*N))

    # Enforce the CFL condition
    if scheme == 'up':
        max_step = h
    elif scheme == 'bw':
        max_step = 2*h
    else:
        raise ValueError('parameter \'scheme\' is not valid!')

    s_eval = np.linspace(0, s_max, num=100)
    sol = solve_ivp(lambda t, y: four_state(t, y, scheme=scheme), [0, s_max], p0,
                    t_eval=s_eval, max_step=max_step)

    fig, ax = plt.subplots()

    u_data, v_data, f_data, vf_data = np.split(sol.y, 4)
    line_u, line_v, line_f, line_vf = ax.plot(
        y, np.append(delta_h(sol.t[0]), u_data[:, 0]), y[1:], v_data[:, 0],
        y[1:], f_data[:, 0], y[1:], vf_data[:, 0])
    line_u.set_label('$q_U$')
    line_v.set_label('$q_V$')
    line_f.set_label('$q_F$')
    line_vf.set_label('$q_{VF}$')
    ax.set_ylim(bottom=-1, top=11)
    vline = ax.axvline(sol.t[0], color='k')
    vline.set_label('$y = t$')
    ax.legend(loc='upper right')
    ax.set_xlabel('$y$')
    ax.set_ylabel('Probability density')

    file_format = 'a{:g}_eps{:g}'.format(a, eps1)

    ani = animation.FuncAnimation(
        fig, animate, frames=sol.t.shape[0], interval=25)
    ani.save('ani_'+file_format+'.mp4')
    plt.show()

    s_min = np.argmax(u_data[-1, :] > 1e-8)
    plt.plot(1 / s_eval[s_min:], s_eval[s_min:]**2*u_data[-1, s_min:])
    plt.axvline(1, color='k')
    plt.xlabel('$v^*$')
    plt.ylabel('Probability density')
    plt.savefig('avg-vel_'+file_format+'.png')
    plt.show()

    # Adiabatic reduction

