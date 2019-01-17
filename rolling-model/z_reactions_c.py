# A simple check with binding in one theta position, several z positions

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import truncnorm
from utils import length


def fixed_z(j, init=None, L=2.5, T=0.4, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
            delta=3.0, kap=1.0, saturation=True, binding='both', seed=None):

    np.random.seed(seed)
    z_vec = np.linspace(-L, L, 2*M+2)
    th_vec = np.linspace(-np.pi/2, np.pi/2, N+2)
    bond_list = np.empty(shape=(0, 1))
    master_list = [bond_list]

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    rate = dt*kap*np.exp(-eta/2*(1 - np.cos(th_vec[j]) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
        erf(np.sqrt(eta/2)*(np.sin(th_vec[j]) + z_vec[-1])) - erf(np.sqrt(eta/2)*(np.sin(th_vec[j]) + z_vec[0]))
    )
    a = (z_vec[0] - th_vec[j])/np.sqrt(1/eta)
    b = (z_vec[-1] - th_vec[j])/np.sqrt(1/eta)

    if init is 'sat':
        bond_list = np.random.uniform(low=-L, high=L, size=bond_max)

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    for k in range(time_steps):
        # Randomly form some bonds
        if on:
            forming_bonds = np.random.poisson(lam=np.maximum(rate*(bond_max - sat*bond_list.shape[0]), 0))
            new_bonds = truncnorm.rvs(a=a, b=b, loc=np.sin(th_vec[j]), scale=np.sqrt(1/eta), size=forming_bonds)

        # Randomly break some bonds
        if off:
            bond_lengths = length(bond_list, th_vec[j], d_prime=d_prime)
            break_probs = np.random.rand(bond_list.shape[0])
            break_indices = np.nonzero(break_probs < (1 - np.exp(
                -dt*np.exp(delta*bond_lengths))))[0]

        # Add and remove the bonds
        bond_list = np.delete(bond_list, break_indices)
        bond_list = np.append(bond_list, new_bonds)

        master_list.append(bond_list)

    return master_list, t


def variable_z(j, init=None, L=2.5, T=0.4, M=100, N=100, bond_max=100, d_prime=0.1, eta=0.1,
               delta=3.0, kap=1.0, saturation=True, binding='both', seed=None):

    np.random.seed(seed)
    z_vec = np.linspace(-L, L, 2*M+2)
    th_vec = np.linspace(-np.pi/2, np.pi/2, N+2)
    bond_list = np.empty(shape=(0, 1))
    master_list = [bond_list]

    t = np.array([0])

    rate = kap*np.exp(-eta/2*(1 - np.cos(th_vec[j]) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
        erf(np.sqrt(eta/2)*(np.sin(th_vec[j]) + z_vec[-1])) - erf(np.sqrt(eta/2)*(np.sin(th_vec[j]) + z_vec[0]))
    )
    a = (z_vec[0] - th_vec[j])/np.sqrt(1/eta)
    b = (z_vec[-1] - th_vec[j])/np.sqrt(1/eta)

    if init is 'sat':
        bond_list = np.random.uniform(low=-L, high=L, size=bond_max)

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    while t[-1] < T:
        break_rates = np.array([])
        form_rates = np.array([])
        # Calculate formation rate
        if on:
            form_rates = np.append(form_rates, rate*(bond_max - sat*bond_list.shape[0]))

        # Calculate breaking rates
        if off:
            bond_lengths = length(bond_list, th_vec[j], d_prime=d_prime)
            break_rates = np.append(break_rates, np.exp(delta*bond_lengths))

        all_rates = np.append(break_rates, form_rates)
        sum_rates = np.cumsum(all_rates)
        a0 = sum_rates[-1]

        r = np.random.rand(2)
        dt = 1/a0*np.log(1/r[0])
        j = np.searchsorted(a=sum_rates, v=r[1]*a0)

        t = np.append(t, t[-1]+dt)

        if j == 0:
            # Pick a binding location
            bond_list = np.append(bond_list, truncnorm.rvs(a=a, b=b, loc=np.sin(th_vec[j]), scale=np.sqrt(1/eta)))
        else:
            # Break the chosen bond
            bond_list = np.delete(bond_list, j-1)

        master_list.append(bond_list)

    return master_list, t


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
trials = 10
time_steps, T = 1000, 2
binding = 'both'
saturation = True

pde, t = pde_z(j, T=T, M=M, time_steps=time_steps, saturation=saturation, binding=binding)

fix_master_list = []
var_master_list = []
t_list = []

for k in range(trials):
    bond_numbers = fixed_z(j, T=T, time_steps=time_steps, bond_max=bond_max,
                          saturation=saturation, binding=binding)[0]
    fix_master_list.append(bond_numbers)
    bond_numbers, var_t = variable_z(j, T=T, bond_max=bond_max, saturation=saturation, binding=binding)
    var_master_list.append(bond_numbers)
    t_list.append(var_t)
    # plt.plot(t, fixed_arr[k, :]/bond_max, t, pde, var_t, np.arange(var_t.shape[0])/bond_max)
    # plt.show()

fix_sto_count = np.zeros(shape=(trials, t.shape[0]))
var_sto_count = np.zeros(shape=(trials, t.shape[0]))
var_sto_count1 = np.zeros(shape=(trials, t.shape[0]))

for i in range(trials):
    for j in range(len(fix_master_list[i])):
        fix_sto_count[i, j] = fix_master_list[i][j].shape[0]

for i in range(trials):
    temp_sto_count = np.zeros(shape=len(var_master_list[i]))
    for j in range(len(var_master_list[i])):
        temp_sto_count[j] = var_master_list[i][j].shape[0]
    var_sto_count[i, :] = temp_sto_count[np.searchsorted(t_list[i], t, side='right')-1]

fixed_avg = np.mean(fix_sto_count, axis=0)
fixed_std = np.std(fix_sto_count, axis=0)
var_avg = np.mean(var_sto_count, axis=0)
var_std = np.std(var_sto_count, axis=0)

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
