import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy.integrate import odeint


def gillespie(y0, trials, proc=4):
    pool = mp.Pool(processes=proc)
    result = [pool.apply_async(gillespie_run, args=(y0,))
              for _ in range(trials)]
    result = [res.get() for res in result]
    return result


def gillespie_run(y0):
    np.random.seed()
    a0, c0, b0 = y0
    a, b, c = np.array([a0]), np.array([b0]), np.array([c0])
    t = np.array([0])
    while t[-1] < T:
        r = np.random.rand(2)
        rates = np.array([kp * a[-1] * b[-1], km * c[-1], kcat * c[-1]])
        cum_rates = np.cumsum(rates, dtype='float64')

        if cum_rates[-1] == 0:
            break

        dt = 1 / cum_rates[-1] * np.log(1 / r[0])
        j = np.searchsorted(cum_rates, r[1] * cum_rates[-1])
        t = np.append(t, t[-1] + dt)
        if j == 0:
            a = np.append(a, a[-1] - 1)
            b = np.append(b, b[-1] - 1)
            c = np.append(c, c[-1] + 1)
        elif j == 1:
            a = np.append(a, a[-1] + 1)
            b = np.append(b, b[-1] + 1)
            c = np.append(c, c[-1] - 1)
        elif j == 2:
            a = np.append(a, a[-1])
            b = np.append(b, b[-1] + 1)
            c = np.append(c, c[-1] - 1)
    return np.array([a, c, b]), t


def dy(y, t):
    a, c, b = y

    da = km*c - kp*a*b
    dc = kp*a*b - (km+kcat)*c
    db = (km+kcat)*c - kp*a*b

    return np.array([da, dc, db])


def next_rxn(y0, trials, proc=4):
    pool = mp.Pool(processes=proc)
    result = [pool.apply_async(nr_run, args=(y0,))
              for _ in range(trials)]
    result = [res.get() for res in result]
    return result


def nr_run(y0):
    np.random.seed()
    a0, c0, b0 = y0

    a, b, c = np.array([a0]), np.array([b0]), np.array([c0])
    t = np.array([0])
    Tk = np.zeros(shape=3)

    rates = np.array([kp*a0*b0, km*c0, kcat*c0])
    del_t = np.zeros(shape=3)
    r = np.random.rand(3)
    P = np.log(1/r)

    while t[-1] < T:
        del_t[rates > 0] = (P - Tk)[rates > 0]/rates[rates > 0]
        del_t[rates==0] = np.inf
        j = np.argmin(del_t)
        delta = del_t[j]

        if delta == np.inf:
            break

        t = np.append(t, t[-1] + delta)
        if j == 0:
            a = np.append(a, a[-1] - 1)
            b = np.append(b, b[-1] - 1)
            c = np.append(c, c[-1] + 1)
        elif j == 1:
            a = np.append(a, a[-1] + 1)
            b = np.append(b, b[-1] + 1)
            c = np.append(c, c[-1] - 1)
        elif j == 2:
            a = np.append(a, a[-1])
            b = np.append(b, b[-1] + 1)
            c = np.append(c, c[-1] - 1)

        Tk += rates*delta
        P[j] += np.log(1/np.random.rand())
        rates = np.array([kp*a[-1]*b[-1], km*c[-1], kcat*c[-1]])

    return np.array([a, c, b]), t


def sample_result(result, t_mesh):
    sampled = list()
    for res in result:
        sampled.append(res[0][:, np.searchsorted(a=res[1], v=t_mesh,
                                                 side='right')-1])
    stacked = np.stack(sampled, axis=2)
    return np.mean(stacked, axis=2)


def dyvar(y, t):
    a, c, b = y

    da = km*np.exp(-R*t)*c - kp*np.exp(-R*t)*a*b
    dc = kp*np.exp(-R*t)*a*b - (km + kcat)*np.exp(-R*t)*c
    db = (km + kcat)*np.exp(-R*t)*c - kp*np.exp(-R*t)*a*b

    return np.array([da, dc, db])


def next_rxn_var(y0, trials, proc=4):
    pool = mp.Pool(processes=proc)
    result = [pool.apply_async(nrv_run, args=(y0,))
              for _ in range(trials)]
    result = [res.get() for res in result]
    return result


def nrv_run(y0):
    np.random.seed()
    a0, c0, b0 = y0

    a, b, c = np.array([a0]), np.array([b0]), np.array([c0])
    t = np.array([0])
    Tk = np.zeros(shape=3)

    rates = np.array([kp*a0*b0, km*c0, kcat*c0])
    del_t = np.zeros(shape=3)
    r = np.random.rand(3)
    P = np.log(1/r)

    while t[-1] < T:
        valid_indices = (P - Tk)*R < rates*np.exp(-R*t[-1])
        del_t[:] = np.inf
        del_t[valid_indices] = -t[-1] - 1/R*np.log(
            np.exp(-R*t[-1]) - R*(P - Tk)[valid_indices]/rates[valid_indices])

        j = np.argmin(del_t)
        delta = del_t[j]

        if delta == np.inf:
            break

        t = np.append(t, t[-1] + delta)
        if j == 0:
            a = np.append(a, a[-1] - 1)
            b = np.append(b, b[-1] - 1)
            c = np.append(c, c[-1] + 1)
        elif j == 1:
            a = np.append(a, a[-1] + 1)
            b = np.append(b, b[-1] + 1)
            c = np.append(c, c[-1] - 1)
        elif j == 2:
            a = np.append(a, a[-1])
            b = np.append(b, b[-1] + 1)
            c = np.append(c, c[-1] - 1)

        Tk += rates/R*(np.exp(-R*t[-2]) - np.exp(-R*t[-1]))
        P[j] += np.log(1/np.random.rand())
        rates = np.array([kp*a[-1]*b[-1], km*c[-1], kcat*c[-1]])

    return np.array([a, c, b]), t


if __name__ == '__main__':
    T = 2
    R = 5
    kp, km, kcat = 1, 5, 5

    y0 = np.array([50, 0, 20], dtype='float64')
    t_mesh = np.linspace(0, T, num=500)

    trials = 10
    gillespie_result = gillespie(y0, trials)
    gillespie_mean = sample_result(gillespie_result, t_mesh)

    nr_result = next_rxn(y0, trials)
    nr_mean = sample_result(nr_result, t_mesh)
    sol = odeint(dy, y0, t_mesh)

    plt.plot(t_mesh, sol[:, 0], label='ODE solution')
    plt.plot(t_mesh, gillespie_mean[0, :], label='Gillespie')
    plt.plot(t_mesh, nr_mean[0, :], label='Next reaction')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of molecules')
    plt.title('Simulated number of A molecules')
    plt.show()

    nrv_result = next_rxn_var(y0, trials)
    nrv_mean = sample_result(nrv_result, t_mesh)
    sol = odeint(dyvar, y0, t_mesh)

    plt.plot(t_mesh, sol[:, 0], label='ODE solution')
    plt.plot(t_mesh, nrv_mean[0, :], label='Next reaction')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of molecules')
    plt.title('Simulated number of A molecules')
    plt.show()
