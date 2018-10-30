import numpy as np
import matplotlib.pyplot as plt
from constructA import length


def pde_only(L=2.5, T=.4, M=100, N=100, time_steps=1000, d_prime=0.1, eta=0.1,
             delta=3.0, kap=1.0, v=18, om=17, saturation=True):

    # Numerical Parameters
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    dt = t[1]-t[0]

    m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))  # Bond densities are initialized to zero
    l_matrix = length(z_mesh, th_mesh, d_prime)

    # Check for the CFL condition

    if om*dt/nu > 1:
        print('Warning: the CFL condition for theta is not satisfied!')
    if v*dt/h > 1:
        print('Warning: the CFL condition for z is not satisfied!')

    for i in range(time_steps):
        if saturation:
            m_mesh[:-1, :-1, i+1] = (m_mesh[:-1, :-1, i] + om*dt/nu*(m_mesh[:-1, 1:, i] - m_mesh[:-1, :-1, i]) +
                                     v*dt/h*(m_mesh[1:, :-1, i] - m_mesh[:-1, :-1, i]) +
                                     dt*kap*np.exp(-eta/2*l_matrix[:-1, :-1]**2) *
                                     (1 - h*np.tile(np.sum(m_mesh[:-1, :-1, i], axis=0), reps=(2*M, 1)))) /\
                                    (1 + dt*np.exp(delta*l_matrix[:-1, :-1]))
        else:
            m_mesh[:-1, :-1, i+1] = (m_mesh[:-1, :-1, i] + om*dt/nu*(m_mesh[:-1, 1:, i] - m_mesh[:-1, :-1, i]) +
                                     v*dt/h*(m_mesh[1:, :-1, i] - m_mesh[:-1, :-1, i]) +
                                     dt*kap*np.exp(-eta/2*l_matrix[:-1, :-1]**2)) /\
                                    (1 + dt*np.exp(delta*l_matrix[:-1, :-1]))

    return m_mesh, t


L, T = 2.5, 0.4
top = 7
m_M, m_N, m_time = list(), list(), list()
t_M, t_N, t_time = list(), list(), list()
M_tot, N_tot, time_tot = list(), list(), list()
err = np.zeros(shape=(top-1, 3))

# Compute the "exact" solution first
m_fine, t_fine = pde_only(L=L, T=T, M=2**top, N=2**top, time_steps=25*2**top)

for j in range(1, top):
    temp_m, temp_t = pde_only(L=L, T=T, M=2**j, N=2**top, time_steps=25*2**top)
    m_M.append(temp_m)
    t_M.append(temp_t)

    temp_m, temp_t = pde_only(L=L, T=T, M=2**top, N=2**j, time_steps=25*2**top)
    m_N.append(temp_m)
    t_N.append(temp_t)

    temp_m, temp_t = pde_only(L=L, T=T, M=2**j, N=2**j, time_steps=25*2**j)
    m_time.append(temp_m)
    t_time.append(temp_t)

for j in range(1, top):
    # Compute the L2 error for each time step
    l2M = L * 2**(-j) * np.sum((m_M[j-1] - m_fine[::2**(top-j), :, :])**2, axis=(0, 1))
    l2N = np.pi * 2**(-j) * np.sum((m_N[j-1] - m_fine[:, ::2**(top-j), :])**2, axis=(0, 1))
    l2time = L * np.pi* 2**(-2*j) * np.sum((m_time[j-1] - m_fine[::2**(top-j), ::2**(top-j), ::2**(top-j)])**2,
                                           axis=(0, 1))

    # Compute the L2 error across time steps
    err[j-1, 0] = np.sqrt(np.sum(l2M[1:] * (t_M[j-1][1:] - t_M[j-1][:-1])))
    err[j-1, 1] = np.sqrt(np.sum(l2N[1:] * (t_N[j-1][1:] - t_M[j-1][:-1])))
    err[j-1, 2] = np.sqrt(np.sum(l2time[1:] * (t_time[j-1][1:] - t_time[j-1][:-1])))

    # Calculate the total number of bonds in each result
    M_tot.append(np.trapz(np.trapz(m_M[j-1], np.linspace(-L, L, 2**(j+1)+1), axis=0),
                          np.linspace(-np.pi/2, np.pi/2, 2**top+1), axis=0))
    N_tot.append(np.trapz(np.trapz(m_N[j-1], np.linspace(-L, L, 2**(top+1)+1), axis=0),
                          np.linspace(-np.pi/2, np.pi/2, 2**j+1), axis=0))
    time_tot.append(np.trapz(np.trapz(m_time[j-1], np.linspace(-L, L, 2**(j+1)+1), axis=0),
                    np.linspace(-np.pi/2, np.pi/2, 2**j+1), axis=0))

steps = 2**np.arange(start=1, stop=top)
q = np.mean(np.log2(err[:-1, :]/err[1:, :]), axis=0)
C = np.max(err*steps[:, None]**q[None, :], axis=0)

print(q)

fig, ax = plt.subplots(ncols=3, figsize=(15, 8))

for j in range(3):
    ax[j].loglog(steps, err[:, j])
    ax[j].loglog(steps, C[j]/steps**q[j], 'k--')
    ax[j].set_xlabel('Steps')
    ax[j].text(steps[1], C[j]/steps[1]**q[j], '$q = {:g}$'.format(q[j]))

ax[0].set_title('Convergence in $h$ ($z$ discretization)')
ax[1].set_title('Convergence in $\\nu$ ($\\theta$ discretization)')
ax[2].set_title('Convergence in $dt$')
for a in ax:
    a.grid(True, which='both')
    a.set_ylabel('Error')

fig1, ax1 = plt.subplots(ncols=3, figsize=(15, 8))
for j in range(1, top):
    ax1[0].plot(t_M[j-1], M_tot[j-1], label='M = {:d}'.format(2**j))
    ax1[1].plot(t_N[j-1], N_tot[j-1], label='N = {:d}'.format(2**j))
    ax1[2].plot(t_time[j-1], time_tot[j-1], label='Time steps = {:d}'.format(25*2**j))
plt.legend()
plt.show()


