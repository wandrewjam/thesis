from constructA import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize


def _up(bond_mesh, vel, dx, dt, axis):
    if axis == 0:
        return vel*dt/dx*(bond_mesh[1:, :-1] - bond_mesh[:-1, :-1])
    elif axis == 1:
        return vel*dt/dx*(bond_mesh[:-1, 1:] - bond_mesh[:-1, :-1])


def _bw(bond_mesh, vel, dx, dt, axis):
    if axis == 0:
        return vel*dt/(2*dx)*(-3*bond_mesh[:-2, :-1] + 4*bond_mesh[1:-1, :-1]
                              - bond_mesh[2:, :-1]) + (vel*dt/dx)**2/2\
               * (bond_mesh[:-2, :-1] - 2*bond_mesh[1:-1, :-1]
                  + bond_mesh[2:, :-1])
    elif axis == 1:
        return vel*dt/(2*dx)*(-3*bond_mesh[:-1, :-2] + 4*bond_mesh[:-1, 1:-1]
                              - bond_mesh[:-1, 2:]) + (vel*dt/dx)**2/2\
               * (bond_mesh[:-1, :-2] - 2*bond_mesh[:-1, 1:-1]
                  + bond_mesh[:-1, 2:])


def _form(bond_mesh, form_rate, h, dt, sat):
    sat_term = (1 - sat*np.tile(np.trapz(bond_mesh[:, :-1], dx=h, axis=0),
                                reps=(bond_mesh.shape[0]-1, 1)))
    return dt*form_rate*sat_term


def _eulerian_step(bond_mesh, v, om, h, nu, dt, form_rate, break_rate, sat,
                   scheme):
    new_bonds = np.copy(bond_mesh)
    if scheme == 'up':
        new_bonds[:-1, :-1] += _up(bond_mesh, v, h, dt, axis=0)
        new_bonds[:-1, :-1] += _up(bond_mesh, om, nu, dt, axis=1)
        new_bonds[:-1, :-1] += _form(bond_mesh, form_rate[:-1, :-1], h, dt,
                                     sat)
        new_bonds[:-1, :-1] /= (1 + dt*break_rate[:-1, :-1])
    elif scheme == 'bw':
        new_bonds[:-2, :-1] += _bw(bond_mesh, v, h, dt, axis=0)
        new_bonds[:-1, :-2] += _bw(bond_mesh, om, nu, dt, axis=1)
        new_bonds[-2, :-1] += np.squeeze(_up(bond_mesh[-2:, :], v, h, dt,
                                             axis=0))
        new_bonds[:-1, -2] += np.squeeze(_up(bond_mesh[:, -2:], om, nu, dt,
                                              axis=1))
        new_bonds[:-1, :-1] += _form(bond_mesh, form_rate[:-1, :-1], h, dt,
                                     sat)
        new_bonds[:-1, :-1] /= (1 + dt*break_rate[:-1, :-1])
    return new_bonds


def time_dependent(expt='unbound', L=2.5, T=.4, M=100, N=100, time_steps=1000,
                   d_prime=0.1, eta=0.1, delta=3.0, kap=1.0, eta_v=0.01,
                   eta_om=0.01, gamma=20.0, sat=True, save_m=False,
                   scheme='up', save_movie=False):
    # Numerical Parameters
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    dt = t[1]-t[0]

    if type(gamma) is float or int:
        om_f = gamma*np.ones(shape=time_steps+1)
    else:
        om_f = gamma
    v_f = (1 + d_prime)*om_f

    om = np.zeros(time_steps + 1)
    v = np.zeros(time_steps + 1)

    if save_m:
        m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))  # Bond densities are initialized to zero
    else:
        m_mesh = np.zeros(shape=(2*M+1, N+1))

    if expt == 'unbound':
        # Initialize platelet velocities to fluid velocities (initial bond density is zero)
        om[0] = om_f[0]
        v[0] = v_f[0]
    elif expt == 'unmoving':
        # Calculate the steady state bond density for an unmoving platelet
        z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')
        A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime, saturation=sat)
        m = spsolve(h*nu*kap*C + nu*D, -R)  # Calculate the bond density function
        if save_m:
            m_mesh[:, :, 0] = m.reshape(2*M+1, -1, order='F')  # Reshape the m vector into an array
        else:
            m_mesh = m.reshape(2*M+1, -1, order='F')
        # Note, the platelet velocities are already initialized to zero
        # plt.pcolormesh(z_mesh, th_mesh, m_mesh[:, :, 0])
        # plt.colorbar()
        # plt.xlabel('$z$ position')
        # plt.ylabel('$\\theta$ position')
        # plt.title('Initial Bond density ($m_0$)')
        # plt.show()

    l_matrix = length(z_mesh, th_mesh, d_prime)
    form_rate = kap*np.exp(-eta/2*l_matrix**2)
    break_rate = np.exp(delta*l_matrix)

    # Check for the CFL condition

    if np.max(om_f)*dt/nu > 1:
        print('Warning: the CFL condition for theta is not satisfied!')
    if np.max(v_f)*dt/h > 1:
        print('Warning: the CFL condition for z is not satisfied!')

    if save_m:
        for i in range(time_steps):
            m_mesh[:, :, i+1] = _eulerian_step(m_mesh[:, :, i], v[i], om[i], h,
                                               nu, dt, form_rate, break_rate,
                                               sat, scheme)

            f_prime = nd_force(m_mesh[:, :, i+1], z_mesh, th_mesh)
            tau = nd_torque(m_mesh[:, :, i+1], z_mesh, th_mesh, d_prime)
            v[i+1], om[i+1] = v_f[i] + f_prime/eta_v, om_f[i] + tau/eta_om
    else:
        for i in range(time_steps):
            m_mesh = _eulerian_step(m_mesh, v[i], om[i], h, nu, dt, form_rate,
                                    break_rate, sat, scheme)

            f_prime = nd_force(m_mesh, z_mesh, th_mesh)
            tau = nd_torque(m_mesh, z_mesh, th_mesh, d_prime)
            v[i+1], om[i+1] = v_f[i] + f_prime/eta_v, om_f[i] + tau/eta_om


    # fig = plt.figure()
    #
    # ims = []
    # norm = Normalize(vmin=0, vmax=np.max(m_mesh))
    # for i in range(time_steps//10+1):
    #     im = plt.pcolormesh(z_mesh, th_mesh, m_mesh[:, :, 10*i], animated=True, norm=norm)
    #     ims.append([im])
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    # plt.colorbar()
    # plt.xlabel('$z$ position')
    # plt.ylabel('$\\theta$ position')
    # plt.title('Bond density ($m$) evolution')
    # plt.show()
    #
    # if save_movie:
    #     ani.save('mov1.mp4')
    #
    # plt.plot(t, v)
    # plt.xlabel('Nondimensional time ($s$)')
    # plt.ylabel('ND linear velocity ($v$)')
    # plt.show()
    #
    # plt.plot(t, om)
    # plt.xlabel('Nondimensional time ($s$)')
    # plt.ylabel('ND angular velocity ($\\omega$)')
    # plt.show()

    return v, om, m_mesh, t


if __name__ == '__main__':
    up_results = time_dependent(M=2**6, N=2**6, time_steps=10*2**6, save_m=True,
                                scheme='up')
    bw_results = time_dependent(M=2**6, N=2**6, time_steps=10*2**6, save_m=True,
                                scheme='bw')

    z_vec = np.linspace(-2.5, 2.5, num=2**7+1)
    th_vec = np.linspace(-np.pi/2, np.pi/2, num=2**6+1)

    up_quant = np.trapz(np.trapz(up_results[2], z_vec, axis=0), th_vec, axis=0)
    bw_quant = np.trapz(np.trapz(bw_results[2], z_vec, axis=0), th_vec, axis=0)

    fig, ax = plt.subplots(ncols=3, figsize=(12, 6))
    ax[0].plot(up_results[-1], up_results[0], label='Upwind scheme')
    ax[0].plot(bw_results[-1], bw_results[0], label='Beam-Warming scheme')
    ax[0].legend()
    ax[0].set_xlabel('Nondimensional time')
    ax[0].set_ylabel('Translation velocity ($v$)')

    ax[1].plot(up_results[-1], up_results[1], label='Upwind scheme')
    ax[1].plot(bw_results[-1], bw_results[1], label='Beam-Warming scheme')
    ax[1].legend()
    ax[1].set_xlabel('Nondimensional time')
    ax[1].set_ylabel('Rotation velocity ($\\omega$)')

    ax[2].plot(up_results[-1], up_quant, label='Upwind scheme')
    ax[2].plot(bw_results[-1], bw_quant, label='Beam-Warming scheme')
    ax[2].legend()
    ax[2].set_xlabel('Nondimensional time')
    ax[2].set_ylabel('Bond quantity')

    plt.tight_layout()
    plt.show()
