import numpy as np
import matplotlib.pyplot as plt
from stochastic_reactions_PDE import pde_motion, pde_bins
from constructA import length
from timeit import default_timer as timer


# Code to analyze convergence of the deterministic schemes with
# prescribed motion

# I need to be more intelligent about choosing a true solution. This
# forcing function is too large in the corners to expect reasonable
# convergence
def _forcing(z_mesh, th_mesh, t, v, om):
    L = 2.5
    kappa, eta = 1, .1
    d, delta = .1, 3
    l_mesh = length(z_mesh, th_mesh, d_prime=d)[:, :]
    prod = (z_mesh - L)*(z_mesh + L + .5)*(th_mesh-np.pi/2)*(th_mesh+np.pi)/10
    integral = 4*t/(1 + 4*t)*(th_mesh-np.pi/2)*(th_mesh + np.pi)/10*(325./12)

    return (-kappa*np.exp(-eta/2*l_mesh**2)*(1 - integral)
            - 16*t*prod/(1 + 4*t)**2 + 4*prod/(1 + 4*t) + 4*t*prod/(1 + 4*t)
            * np.exp(delta*l_mesh) - 4*v*t/(1 + 4*t)*(th_mesh - np.pi/2)
            * (th_mesh + np.pi)/10 * (2*z_mesh + 0.5)
            - 4*om*t/(1 + 4*t)*(z_mesh - L)*(z_mesh + L + .5)/10*(th_mesh
                                                                  + np.pi/2))


def _true(z_mesh, th_mesh, t):
    prod = (z_mesh - 2.5)*(z_mesh + 3)*(th_mesh - np.pi/2)*(th_mesh + np.pi)/10
    return 4*t*prod/(1 + 4*t)


def _err_estimate(approx, true, L=2.5, T=1, bins=False):
    z_steps, th_steps, t_steps = approx.shape
    z_mesh = np.linspace(-L, L, num=z_steps)
    t_mesh = np.linspace(0, T, num=t_steps)
    if bins:
        nu = 2*np.pi/th_steps
        return np.sqrt(
            np.trapz(nu*np.sum(np.trapz((approx-true)**2, z_mesh, axis=0),
                               axis=0), t_mesh)
        )
    else:
        th_mesh = np.linspace(-np.pi/2, np.pi/2, num=th_steps)
        return np.sqrt(
            np.trapz(np.trapz(np.trapz((approx-true)**2, z_mesh, axis=0),
                              th_mesh, axis=0), t_mesh)
        )


def conv_study(v, om, top, L=2.5, T=1):
    exponents = np.arange(top) + 1
    Ms, Ns = 2**exponents, 2**exponents
    steps = Ms/2*np.maximum(v, om)

    def forcing(z_mesh, th_mesh, t):
        return _forcing(z_mesh[:, :, None], th_mesh[:, :, None], t[None, None, :], v, om)

    up_results, bw_results, bn_results = list(), list(), list()
    for i in range(top):
        up_results.append(pde_motion(v, om, T=T, M=Ms[i], N=Ns[i],
                                     time_steps=steps[i], scheme='up')[2])
        bw_results.append(pde_motion(v, om, T=T, M=Ms[i], N=Ns[i],
                                     time_steps=steps[i], scheme='bw')[2])
        bn_results.append(pde_bins(v, om, T=T, M=Ms[i], N=Ns[i],
                                   time_steps=steps[i])[2])

    up_errs, bw_errs, bn_errs = list(), list(), list()
    for i in range(top-1):
        skip = 2**(top - i - 1)

        up_errs.append(
            _err_estimate(up_results[i], up_results[-1][::skip, ::skip, ::skip],
                          L, T)
        )
        bw_errs.append(
            _err_estimate(bw_results[i], bw_results[-1][::skip, ::skip, ::skip],
                          L, T)
        )
        bn_errs.append(
            _err_estimate(bn_results[i], bn_results[-1][::skip, ::skip, ::skip],
                          L, T, bins=True)
        )

    return 1./Ms, up_errs, bw_errs, bn_errs


if __name__ == '__main__':
    in_str = raw_input('Enter v, om, and top: ')
    in_str = in_str.split()
    v, om = float(in_str[0]), float(in_str[1])
    top = int(in_str[2])

    steps, up_errs, bw_errs, bn_errs = conv_study(v, om, top)

    steps = steps[:-1]
    plt.loglog(steps, up_errs, label='Upwind scheme')
    plt.loglog(steps, bw_errs, label='Beam-Warming scheme')
    plt.loglog(steps, bn_errs, label='Semi-lagrangian scheme')
    plt.legend()
    plt.show()

    np.savez_compressed('./data/conv_data/det_schemes_conv1.npz')
