# A simple check with binding in one theta position, several z positions

import numpy as np
import matplotlib.pyplot as plt
from constructA import length


def fixed_z(j, init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
            delta=3.0, kap=1.0, saturation=True, binding='both', seed=None):

    np.random.seed(seed)
    z_vec = np.linspace(-L, L, 2*M+2)
    th_vec = np.linspace(-np.pi/2, np.pi, N+2)
    z_mesh, th_mesh = np.meshgrid(z_vec, th_vec, indexing='ij')
    bond_number = np.zeros(shape=(2*M+2, time_steps+1))
    h = z_vec[1] - z_vec[0]

    if init is 'sat':
        bond_number = bond_max

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
    rate = (h*kap*np.exp(-eta/2*l_matrix**2))[:, j]
    b_rate = (np.exp(delta*l_matrix))[:, j]
    correction = np.ones(shape=2*M+2)
    correction[[0, -1]] = 1/2

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    for k in range(time_steps):
        # Randomly form some bonds
        bond_number[:, k+1] = bond_number[:, k] + on*np.random.poisson(
            lam=np.maximum(dt*correction*rate*(bond_max - sat*np.sum(bond_number[:, k])), 0))
        # Randomly break some bonds
        bond_number[:, k+1] = np.maximum(bond_number[:, k+1] -
                                         off*np.random.poisson(lam=dt*bond_number[:, k]*b_rate), 0)

    return bond_number, t


def variable_z(j, init=None, L=2.5, T=0.4, M=100, N=100, bond_max=100, d_prime=0.1, eta=0.1,
               delta=3.0, kap=1.0, saturation=True, binding='both', seed=None):

    np.random.seed(seed)
    z_vec = np.linspace(-L, L, 2*M+2)
    th_vec = np.linspace(-np.pi/2, np.pi, N+2)
    z_mesh, th_mesh = np.meshgrid(z_vec, th_vec, indexing='ij')
    bond_number = [np.zeros(shape=2*M+2)]
    h = z_vec[1] - z_vec[0]

    if init is 'sat':
        bond_number[:] = bond_max

    t = np.array([0])

    l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
    rate = (h*kap*np.exp(-eta/2*l_matrix**2))[:, j]
    b_rate = (np.exp(delta*l_matrix))[:, j]
    correction = np.ones(shape=2*M+2)
    correction[[0, -1]] = 1/2

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    while t[-1] < T:
        r = np.random.rand(2)
        all_rates = np.append(on*rate*(bond_max - sat*np.sum(bond_number[-1])),
                              off*b_rate*bond_number[-1])
        sum_rates = np.cumsum(all_rates)

        if sum_rates[-1] == 0:
            break

        dt = np.log(1/r[0])/sum_rates[-1]
        t = np.append(t, t[-1] + dt)

        i = np.searchsorted(a=sum_rates, v=r[1]*sum_rates[-1])
        if i < 2*M+2:
            new_bonds = np.copy(bond_number[-1])
            new_bonds[i] += 1
            bond_number.append(new_bonds)
        else:
            new_bonds = bond_number[-1]
            new_bonds[i - (2*M+2)] -= 1
            bond_number.append(new_bonds)

    return bond_number, t


def pde_z(j, init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, d_prime=0.1, eta=0.1,
          delta=3.0, kap=1.0, saturation=True, binding='both'):

    z_vec = np.linspace(-L, L, 2*M+2)
    th_vec = np.linspace(-np.pi/2, np.pi, N+2)
    z_mesh, th_mesh = np.meshgrid(z_vec, th_vec, indexing='ij')
    h = z_vec[1] - z_vec[0]
    nu = th_vec[1] - th_vec[0]
    bond_number = np.zeros(shape=(2*M+2, time_steps+1))

    if init is 'sat':
        bond_number = 1

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
    rate = (kap*np.exp(-eta/2*l_matrix**2))[:, j]
    b_rate = (np.exp(delta*l_matrix))[:, j]

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    for k in range(time_steps):
        bond_number[:, k + 1] = (bond_number[:, k] + on*dt*rate*(1 - sat*np.trapz(bond_number[:, k], z_vec)))\
                                / (1 + off*dt*b_rate)
    return bond_number, t


M = 2**6
j = 51
bond_max = 100
trials = 100
time_steps, T = 1000, 2
binding = 'both'
saturation = True

pde, t = pde_z(j, T=T, M=M, time_steps=time_steps, saturation=saturation, binding=binding)

fixed_arr = np.zeros(shape=(trials, time_steps+1))
var_arr = np.zeros(shape=(trials, time_steps+1))
for k in range(trials):
    bond_number = fixed_z(j, T=T, M=M, time_steps=time_steps, bond_max=bond_max,
                          saturation=saturation, binding=binding)[0]
    fixed_arr[k, :] = np.sum(bond_number, axis=0)
    bond_number, var_t = variable_z(j, T=T, M=M, bond_max=bond_max, saturation=saturation, binding=binding)
    temp_count = np.zeros(shape=len(bond_number))
    for i in range(len(bond_number)):
        temp_count[i] = np.sum(bond_number[i])
    var_arr[k, :] = temp_count[np.searchsorted(var_t, t, side='right')-1]
    # plt.plot(t, fixed_arr[k, :]/bond_max, t, pde, var_t, np.arange(var_t.shape[0])/bond_max)
    # plt.show()

fixed_avg = np.mean(fixed_arr, axis=0)
var_avg = np.mean(var_arr, axis=0)
fixed_std = np.std(fixed_arr, axis=0)
var_std = np.std(var_arr, axis=0)

determ = np.trapz(pde, np.linspace(-2.5, 2.5, 2*M+2), axis=0)

plt.plot(t[1:], (fixed_avg/bond_max - determ)[1:]/determ[1:], 'b', label='Fixed time step')
plt.plot(t[1:], (var_avg/bond_max - determ)[1:]/determ[1:], 'g', label='Variable time step')
plt.plot(t[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))/bond_max - determ)[1:]/determ[1:], 'b:',
         t[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))/bond_max - determ)[1:]/determ[1:], 'b:', linewidth=.5)
plt.plot(t[1:], ((var_avg + 2*var_std/np.sqrt(trials))/bond_max - determ)[1:]/determ[1:], 'g:',
         t[1:], ((var_avg - 2*var_std/np.sqrt(trials))/bond_max - determ)[1:]/determ[1:], 'g:', linewidth=.5)
plt.plot(t, np.zeros(shape=t.shape), 'k')
plt.legend(loc='best')
plt.title('Relative errors for the Stochastic Algorithms for multiple $z$')
plt.show()
