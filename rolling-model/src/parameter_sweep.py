import multiprocessing as mp
import numpy as np
from scipy.integrate import trapz
from timeit import default_timer as timer


def z_fn(s, z_0):
    return z_0 - np.pi*s


def th_fn(s):
    return np.pi/2 - np.pi*s


def on(length, kappa, eta, delta):
    return kappa*np.exp(-eta/2*length**2)/np.sqrt(2*np.pi/eta)


def off(length, kappa, eta, delta):
    return np.exp(delta*length)


def solve_along_chars(z_0, omega, kappa, eta, delta):
    start = timer()
    s_vec = np.linspace(0, 1, num=1001)
    h = s_vec[1] - s_vec[0]
    s_vec = s_vec[:, None, None]
    z_0 = z_0[None, :, None]
    omega = omega[None, None, :]

    len_arr = np.sqrt((1 - np.cos(th_fn(s_vec)))**2 + (np.sin(th_fn(s_vec)) - z_fn(s_vec, z_0))**2)
    tau_arr = -((1 - np.cos(th_fn(s_vec)))*np.sin(th_fn(s_vec))
                + (np.sin(th_fn(s_vec)) - z_fn(s_vec, z_0))*np.cos(th_fn(s_vec)))

    on_arr, off_arr = on(len_arr, kappa, eta, delta), off(len_arr, kappa, eta, delta)
    m_arr = np.zeros(shape=(s_vec.shape[0], z_0.shape[1], omega.shape[2]))

    for i in range(0, m_arr.shape[0]-1):
        m_arr[i+1] = ((m_arr[i] - np.pi*h/(2*omega)*(off_arr[i]*m_arr[i] - on_arr[i] - on_arr[i+1]))
                      / (1 + np.pi*h*off_arr[i+1]/(2*omega)))
    M = np.pi*trapz(trapz(m_arr, z_0, axis=1), dx=h, axis=0)
    T = np.pi*trapz(trapz(tau_arr*m_arr, z_0, axis=1), dx=h, axis=0)
    end = timer()
    print(end-start)
    return M, T


def parameter_sweep(kap_vec, eta_vec, del_vec, proc=4):
    z_0 = np.linspace(-5, 5, num=1001)
    omegas = np.linspace(0, 200, num=101)[1:]

    # Use multiprocessing to run this in parallel, there are probably
    # some numpy functions to manipulate these arrays so I can run
    # through them in a single loop

    pool = mp.Pool(processes=proc)

    result = [
        pool.apply_async(solve_along_chars,
                         args=(z_0, omegas, kappa, eta, delta))
        for kappa in kap_vec for eta in eta_vec for delta in del_vec]
    result = [res.get() for res in result]

    M = [res[0] for res in result]
    T = [res[1] for res in result]

    M = np.reshape(M, newshape=(kap_vec.shape + eta_vec.shape
                                + del_vec.shape + omegas.shape))
    T = np.reshape(T, newshape=(kap_vec.shape + eta_vec.shape
                                + del_vec.shape + omegas.shape))

    return omegas, M, T


if __name__ == '__main__':
    N = 2**6 + 1
    kap_vec = np.logspace(-3, 2, num=N)
    eta_vec = np.logspace(2, 6, num=N)
    del_vec = np.logspace(-1, 1, num=N)
    kap_msh, kap_mid = kap_vec[::2], kap_vec[1::2]
    eta_msh, eta_mid = eta_vec[::2], eta_vec[1::2]
    del_msh, del_mid = del_vec[::2], del_vec[1::2]
    start = timer()
    omegas, M, T = parameter_sweep(kap_mid, eta_mid, del_mid, proc=64)
    end = timer()

    print('Total time required: {} seconds'.format(end-start))
    np.savez_compressed('../data/toy-model/parameter_sweep_coarse1.npz', M, T,
                        kap_vec, eta_vec, del_vec, omegas, M=M, T=T,
                        kap_vec=kap_vec, eta_vec=eta_vec, del_vec=del_vec,
                        omegas=omegas)
