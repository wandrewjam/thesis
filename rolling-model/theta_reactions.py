# A simple check with binding in several theta positions, one z position for each theta

import numpy as np
import matplotlib.pyplot as plt
from utils import length


def fixed_theta(init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
                delta=3.0, kap=1.0, saturation=True, binding='both', seed=None):

    np.random.seed(seed)
    th_vec = np.linspace(-np.pi/2, np.pi/2, N+2)
    bond_number = np.zeros(shape=(N+2, time_steps+1))

    if init is 'sat':
        bond_number[:, 0] = bond_max

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    l_matrix = length(np.sin(th_vec), th_vec, d_prime=d_prime)
    rate = kap*np.exp(-eta/2*l_matrix**2)
    b_rate = np.exp(delta*l_matrix)

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    for k in range(time_steps):
        # Randomly form some bonds
        bond_number[:, k+1] = bond_number[:, k] + on*np.minimum(
            np.random.poisson(lam=dt*rate*(bond_max - sat*bond_number[:, k])), bond_max - bond_number[:, k])
        # Randomly break some bonds
        bond_number[:, k+1] = np.maximum(bond_number[:, k+1] - off*np.random.poisson(lam=dt*bond_number[:, k]*b_rate), 0)
    return bond_number, t


def variable_theta(init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
                   delta=3.0, kap=1.0, saturation=True, binding='both', seed=None):

    np.random.seed(seed)
    th_vec = np.linspace(-np.pi/2, np.pi/2, N+2)
    bond_number = [np.zeros(shape=N+2)]

    if init is 'sat':
        bond_number[:, 0] = bond_max

    t = [0]

    l_matrix = length(np.sin(th_vec), th_vec, d_prime=d_prime)
    rate = kap*np.exp(-eta/2*l_matrix**2)
    b_rate = np.exp(delta*l_matrix)

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    while t[-1] < T:
        r = np.random.rand(2)
        all_rates = np.append(on*rate*(bond_max - sat*bond_number[-1]), off*b_rate*bond_number[-1])
        sum_rates = np.cumsum(all_rates)

        dt = np.log(1/r[0])/sum_rates[-1]
        t = np.append(t, t[-1] + dt)

        j = np.searchsorted(a=sum_rates, v=r[1]*sum_rates[-1])
        if j < N+2:
            new_bonds = np.copy(bond_number[-1])
            new_bonds[j] += 1
            bond_number.append(new_bonds)
        else:
            new_bonds = bond_number[-1]
            new_bonds[j - (N+2)] -= 1
            bond_number.append(new_bonds)

    return bond_number, t


def pde_theta(init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, d_prime=0.1, eta=0.1,
               delta=3.0, kap=1.0, saturation=True, binding='both'):

    th_vec = np.linspace(-np.pi/2, np.pi/2, N+2)
    bond_number = np.zeros(shape=(N+2, time_steps+1))

    if init is 'sat':
        bond_number[:] = 1

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    l_matrix = length(np.sin(th_vec), th_vec, d_prime=d_prime)
    rate = kap*np.exp(-eta/2*l_matrix**2)
    b_rate = np.exp(delta*l_matrix)

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    for k in range(time_steps):
        bond_number[:, k + 1] = (bond_number[:, k] + on*dt*rate*(1 - sat*bond_number[:, k]))/(1 + off*dt*b_rate)
    return bond_number, t


bond_max = 100
trials = 100
time_steps, T = 1000, 1
binding = 'both'
saturation = True

pde, t = pde_theta(T=T, time_steps=time_steps, saturation=saturation, binding=binding)

fixed_arr = np.zeros(shape=(trials, time_steps+1))
var_arr = np.zeros(shape=(trials, time_steps+1))
for k in range(trials):
    bond_number = fixed_theta(T=T, time_steps=time_steps, bond_max=bond_max,
                              saturation=saturation, binding=binding)[0]
    fixed_arr[k, :] = np.sum(bond_number, axis=0)
    bond_number, var_t = variable_theta(T=T, bond_max=bond_max, saturation=saturation, binding=binding)
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

plt.plot(t, fixed_avg/bond_max, 'b', label='Fixed time step')
plt.plot(t, np.sum(pde, axis=0), 'r', label='ODE solution')
plt.plot(t, var_avg/bond_max, 'g', label='Variable time step')
plt.plot(t, (fixed_avg + 2*fixed_std/np.sqrt(trials))/bond_max, 'b:',
         t, (fixed_avg - 2*fixed_std/np.sqrt(trials))/bond_max, 'b:', linewidth=.5)
plt.plot(t, (var_avg + 2*var_std/np.sqrt(trials))/bond_max, 'g:',
         t, (var_avg - 2*var_std/np.sqrt(trials))/bond_max, 'g:', linewidth=.5)
plt.legend(loc='best')
plt.title('Deterministic and Stochastic Algorithms for multiple $\\theta$')
plt.show()
