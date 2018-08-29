from constructA import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize


def time_dependent(expt='unbound', save_movie=False, L=2.5, T=.4, M=100, N=100, time_steps=1000,
                   d_prime=0.1, eta=0.1, delta=3.0, kap=1.0, eta_v=0.01, eta_om=0.01, gamma=20.0, saturation=True):
    # Numerical Parameters
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    dt = t[1]-t[0]
    lam = nu/h

    om_f = gamma
    v_f = (1 + d_prime)*gamma

    om = np.zeros(time_steps + 1)
    v = np.zeros(time_steps + 1)

    m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))  # Bond densities are initialized to zero

    if expt == 'unbound':
        # Initialize platelet velocities to fluid velocities (initial bond density is zero)
        om[0] = om_f
        v[0] = v_f
    elif expt == 'unmoving':
        # Calculate the steady state bond density for an unmoving platelet
        z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')
        A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime)
        m = spsolve(h*nu*kap*C + nu*D, -R)  # Calculate the bond density function
        m_mesh[:, :, 0] = m.reshape(2*M+1, -1, order='F')  # Reshape the m vector into an array
        # Note, the platelet velocities are already initialized to zero
        plt.pcolormesh(z_mesh, th_mesh, m_mesh[:, :, 0])
        plt.colorbar()
        plt.xlabel('$z$ position')
        plt.ylabel('$\\theta$ position')
        plt.title('Initial Bond density ($m_0$)')
        plt.show()

    l_matrix = length(z_mesh, th_mesh, d_prime)

    # Check for the CFL condition

    if om_f*dt/nu > 1:
        print('Warning: the CFL condition for theta is not satisfied!')
    if v_f*dt/h > 1:
        print('Warning: the CFL condition for z is not satisfied!')

    for i in range(time_steps):
        if saturation:
            m_mesh[:-1, :-1, i+1] = (m_mesh[:-1, :-1, i] + om[i]*dt/nu*(m_mesh[:-1, 1:, i] - m_mesh[:-1, :-1, i]) +
                                     v[i]*dt/h*(m_mesh[1:, :-1, i] - m_mesh[:-1, :-1, i]) +
                                     dt*kap*np.exp(-eta/2*l_matrix[:-1, :-1]**2) *
                                     (1 - h*np.tile(np.sum(m_mesh[:-1, :-1, i], axis=0), reps=(2*M, 1)))) /\
                                    (1 + dt*np.exp(delta*l_matrix[:-1, :-1]))
        else:
            m_mesh[:-1, :-1, i+1] = (m_mesh[:-1, :-1, i] + om[i]*dt/nu*(m_mesh[:-1, 1:, i] - m_mesh[:-1, :-1, i]) +
                                     v[i]*dt/h*(m_mesh[1:, :-1, i] - m_mesh[:-1, :-1, i]) +
                                     dt*kap*np.exp(-eta/2*l_matrix[:-1, :-1]**2)) /\
                                    (1 + dt*np.exp(delta*l_matrix[:-1, :-1]))
        f_prime = nd_force(m_mesh[:-1, :-1, i+1], z_mesh[:-1, :-1], th_mesh[:-1, :-1])
        tau = nd_torque(m_mesh[:-1, :-1, i+1], z_mesh[:-1, :-1], th_mesh[:-1, :-1], d_prime)
        v[i+1], om[i+1] = v_f + f_prime/eta_v, om_f + tau/eta_om

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

    return v, om, t, m_mesh
