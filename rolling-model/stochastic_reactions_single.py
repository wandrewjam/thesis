# A simple check with binding in one theta position, several z positions

import numpy as np
import matplotlib.pyplot as plt
from constructA import length
from timeit import default_timer as timer
from scipy.special import erf
from scipy.stats import truncnorm
from time import strftime


def fixed_z(theta, init=None, L=2.5, T=0.4, M=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
            delta=3.0, kap=1.0, saturation=True, binding='both', ztype='cont_exact', seed=None):

    np.random.seed(seed)
    z_vec = np.linspace(-L, L, 2*M+1)[:-1]
    h = z_vec[1] - z_vec[0]
    z_vec += h/2

    if init is None:
        bond_list = np.empty(shape=(0, 1))
    elif init is 'sat':
        bond_list = np.random.uniform(low=-L, high=L, size=(bond_max, 1))
    else:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1
    master_list = [bond_list]

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    if ztype is 'cont_exact':
        # For continuous z, exact rate integral
        expected_coeffs = dt*kap*np.exp(-eta/2*(1 - np.cos(theta) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(theta) + L)) - erf(np.sqrt(eta/2)*(np.sin(theta) - L))
        )
        a = (-L - np.sin(theta))/np.sqrt(1/eta)
        b = (L - np.sin(theta))/np.sqrt(1/eta)
    elif ztype is 'cont_approx':
        # For continuous z, approximate rate integral
        l_matrix = length(z_vec, theta, d_prime=d_prime)
        expected_coeffs = dt*kap*np.trapz(np.exp(-eta/2*l_matrix**2), z_vec, axis=0)  # I'm not sure about this
        # expected_coeffs = dt*kap*h*np.sum(np.exp(-eta/2*l_matrix**2), axis=0)
        a = (-L - theta)/np.sqrt(1/eta)
        b = (L - theta)/np.sqrt(1/eta)
    elif ztype is 'discrete':
        # For discrete z
        l_matrix = length(z_vec, theta, d_prime=d_prime)
        expected_coeffs = h*dt*kap*np.exp(-eta/2*l_matrix**2)

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    for k in range(time_steps):
        # Decide which bonds break
        bond_lengths = length(bond_list[:, 0], theta, d_prime=d_prime)
        break_probs = np.random.rand(bond_list.shape[0])
        break_indices = np.nonzero(break_probs < (1 - np.exp(
            -dt*off*np.exp(delta*bond_lengths))))[0]

        # Decide which bonds form
        bond_count = bond_list.shape[0]
        if ztype is 'cont_exact' or ztype is 'cont_approx':
            # Continuous z formulation
            expected_vals = on*expected_coeffs*(bond_max - sat*bond_count)  # Here expected_coeffs is a scalar
        elif ztype is 'discrete':
            # Discrete z formulation
            expected_vals = on*expected_coeffs*(bond_max - sat*bond_count)  # Here expected_coeffs is a vector

        # Generate bonds in each bin
        if ztype is 'cont_exact' or ztype is 'cont_approx':
            # Continuous z formulation
            expected_vals = np.amin(expected_vals, 0)
            forming_bonds = np.random.poisson(lam=expected_vals)

            new_bonds = np.zeros(shape=(np.sum(forming_bonds), 1))
            counter = 0

            # Choose lengths, add to new bond array
            new_bonds[counter:counter+forming_bonds, 0] = truncnorm.rvs(a=a, b=b, loc=np.sin(theta),
                                                                           scale=np.sqrt(1/eta),
                                                                           size=forming_bonds)
            counter += forming_bonds
        elif ztype is 'discrete':
            # Discrete z formulation
            expected_vals = expected_vals.clip(min=0)
            forming_bonds = np.random.poisson(lam=expected_vals)

            nonzeros = np.transpose(forming_bonds.nonzero())
            new_bonds = np.zeros(shape=nonzeros.shape)
            counter = 0

            for j, k in nonzeros:
                new_bonds[counter:counter+forming_bonds[j, k]] = [[z_vec[j]]]
                counter += forming_bonds[j, k]

        # Update the bond array
        if binding is 'both' or binding is 'off':
            bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        if binding is 'both' or binding is 'on':
            bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Store the bond list
        master_list.append(bond_list)
    return master_list, t


def variable_z(theta, init=None, L=2.5, T=0.4, M=100, bond_max=100, d_prime=0.1, eta=0.1, delta=3.0, kap=1.0,
               saturation=True, binding='both', ztype='cont_exact', seed=None):

    np.random.seed(seed)
    z_vec = np.linspace(-L, L, 2*M+1)[:-1]
    h = z_vec[1] - z_vec[0]
    z_vec += h/2

    if init is None:
        bond_list = np.empty(shape=(0, 1))
    elif init is 'sat':
        bond_list = np.random.uniform(low=-L, high=L, size=(bond_max, 1))
    else:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1
    master_list = [bond_list]

    t = np.array([0])

    if ztype is 'cont_exact':
        # For continuous z, exact rate integral
        expected_coeffs = kap*np.exp(-eta/2*(1 - np.cos(theta) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(theta) + L)) - erf(np.sqrt(eta/2)*(np.sin(theta) - L))
        )
        a = (-L - np.sin(theta))/np.sqrt(1/eta)
        b = (L - np.sin(theta))/np.sqrt(1/eta)
    elif ztype is 'cont_approx':
        # For continuous z, approximate rate integral
        l_matrix = length(z_vec, theta, d_prime=d_prime)
        expected_coeffs = kap*np.trapz(np.exp(-eta/2*l_matrix**2), z_vec, axis=0)  # I'm not sure about this
        # expected_coeffs = dt*kap*h*np.sum(np.exp(-eta/2*l_matrix**2), axis=0)
        a = (-L - theta)/np.sqrt(1/eta)
        b = (L - theta)/np.sqrt(1/eta)
    elif ztype is 'discrete':
        # For discrete z
        l_matrix = length(z_vec, theta, d_prime=d_prime)
        expected_coeffs = h*kap*np.exp(-eta/2*l_matrix**2)

    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'off')
    sat = saturation

    while t[-1] < T:
        # Decide which bonds break
        bond_lengths = length(bond_list[:, 0], theta, d_prime=d_prime)
        break_rates = off*np.exp(delta*bond_lengths)

        # Decide which bonds form
        bond_counts = bond_list.shape[0]
        if ztype is 'cont_exact' or ztype is 'cont_approx':
            # Continuous z formulation
            form_rates = on*expected_coeffs*(bond_max - sat*bond_counts)  # Calculate the expected values
        elif ztype is 'discrete':
            # Discrete z formulation
            form_rates = on*expected_coeffs*(bond_max - sat*bond_counts)

        # Generate bonds in each bin
        if ztype is 'cont_exact' or ztype is 'cont_approx':
            # Continuous z formulation
            all_rates = np.append(break_rates, form_rates)
            sum_rates = np.cumsum(all_rates)
            a0 = sum_rates[-1]

            r = np.random.rand(2)
            dt = 1/a0*np.log(1/r[0])
            j = np.searchsorted(a=sum_rates, v=r[1]*a0)
            t = np.append(t, t[-1]+dt)

            if j < break_rates.shape[0]:
                bond_list = np.delete(arr=bond_list, obj=j, axis=0)
            else:
                index = j - break_rates.shape[0]
                new_bonds = np.zeros(shape=(1, 1))
                new_bonds[0, 0] = truncnorm.rvs(a=a, b=b, loc=np.sin(theta), scale=np.sqrt(1/eta))
                bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)
        elif ztype is 'discrete':
            # Discrete z formulation
            all_rates = np.append(break_rates, form_rates)
            sum_rates = np.cumsum(all_rates)
            if sum_rates[-1] == 0:
                a0 = 1e-10
            else:
                a0 = sum_rates[-1]

            r = np.random.rand(2)
            dt = 1/a0*np.log(1/r[0])
            j = np.searchsorted(a=sum_rates, v=r[1]*a0)
            t = np.append(t, t[-1] + dt)

            if j < break_rates.shape[0]:
                bond_list = np.delete(arr=bond_list, obj=j, axis=0)
            else:
                index = j - break_rates.shape[0]
                new_bonds = np.zeros(shape=(1, 1))
                new_bonds[0, 0] = z_vec[index]
                bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Store the bond list
        master_list.append(bond_list)
    return master_list, t


def pde_z(theta, init=None, L=2.5, T=.4, M=100, time_steps=1000, d_prime=0.1, eta=0.1, delta=3.0,
          kap=1.0, saturation=True, binding='both'):

    z_vec = np.linspace(-L, L, 2*M+1)
    h = z_vec[1] - z_vec[0]
    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    m_mesh = np.zeros(shape=(2*M+1, time_steps+1))  # Bond densities are initialized to zero
    if init is 'sat':
        m_mesh[:, :, 0] = 1/(2*L)
    elif init is not None:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1

    l_matrix = length(z_vec, theta, d_prime)

    on = binding is 'both' or binding is 'on'
    off = binding is 'both' or binding is 'off'
    sat = saturation
    for i in range(time_steps):
        m_mesh[:, i+1] = (m_mesh[:, i] + on*dt*kap*np.exp(-eta/2*l_matrix**2) *
                             (1 - sat*np.tile(np.trapz(y=m_mesh[:, i], x=z_vec,
                                                       axis=0), reps=(2*M+1)))) /\
                             (1 + off*dt*np.exp(delta*l_matrix))

    return z_vec, m_mesh, t


if __name__ == '__main__':
    trials = int(raw_input('Number of trials: '))
    L = 2.5
    M = int(raw_input('M: '))
    theta = 0
    bond_max = 100
    time_steps = int(raw_input('time steps: '))
    T = 2
    init = None
    binding = 'both'
    sat = True
    ztype = 'cont_exact'
    delta = 3

    plot = raw_input('Do you want to plot results? y or n: ')
    while plot != 'y' and plot != 'n':
        plot = raw_input('Please enter \'y\' or \'n\'. Do you want to plot results? ')

    z_vec, m_mesh, tp = pde_z(theta, init=init, T=T, M=M, time_steps=time_steps, delta=delta,
                              saturation=sat, binding=binding)

    fixed_arr = np.zeros(shape=(trials, time_steps+1))
    var_arr = np.zeros(shape=(trials, time_steps+1))

    for k in range(trials):
        start = timer()
        temp_list = fixed_z(theta, init=init, T=T, M=M, time_steps=time_steps, bond_max=bond_max, delta=delta,
                            saturation=sat, binding=binding, ztype=ztype)[0]
        for j in range(len(temp_list)):
            fixed_arr[k, j] = temp_list[j].shape[0]

        temp_list, var_t = variable_z(theta, init=init, T=T, M=M, bond_max=bond_max, delta=delta,
                                      saturation=sat, binding=binding, ztype=ztype)
        temp_count = np.zeros(shape=len(temp_list))
        for j in range(len(temp_list)):
            temp_count[j] = temp_list[j].shape[0]
        var_arr[k, :] = temp_count[np.searchsorted(var_t, tp, side='right')-1]
        end = timer()
        print('{:d} of {:d} trials completed. This run took {:g} seconds.'.format(k+1, trials, end-start))
        # plt.plot(t, fixed_arr[k, :]/bond_max, t, pde, var_t, np.arange(var_t.shape[0])/bond_max)
        # plt.show()

    fixed_avg = np.mean(fixed_arr, axis=0)
    var_avg = np.mean(var_arr, axis=0)
    fixed_std = np.std(fixed_arr, axis=0)
    var_std = np.std(var_arr, axis=0)

    pde_count = np.trapz(m_mesh, z_vec, axis=0)

    # Define parameter array and filename, and save the count data
    # The time is included to prevent overwriting an existing file
    par_array = np.array([delta, T, init, sat, binding, M, bond_max, L, ztype])
    file_path = './data/sta_rxns/'
    file_name = 'singlerec_M{0:d}_theta{1:g}_trials{2:d}_{3:s}.npz'.format(M, theta, trials, strftime('%d%m%y'))
    np.savez_compressed(file_path+file_name, par_array, fixed_arr, var_arr, pde_count, tp, par_array=par_array,
                        fixed_array=fixed_arr, var_array=var_arr, pde_count=pde_count, tp=tp)
    print('Data saved in file {:s}'.format(file_name))

    if plot == 'y':
        plt.plot(tp[1:], (fixed_avg/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed time step')
        plt.plot(tp[1:], (var_avg/bond_max - pde_count)[1:]/pde_count[1:], 'g', label='Variable time step')
        plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'b:',
                 tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'b:', linewidth=.5)
        plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'g:',
                 tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'g:', linewidth=.5)
        plt.plot(tp, np.zeros(shape=tp.shape), 'k')
        plt.legend(loc='best')
        plt.title('Relative errors for the stochastic algorithms for a single receptor')
        plt.show()

        plt.plot(tp, fixed_avg/bond_max, 'b', label='Fixed time step')
        plt.plot(tp, var_avg/bond_max, 'g', label='Variable time step')
        plt.plot(tp, (fixed_avg + 2*fixed_std/np.sqrt(trials))/bond_max, 'b:',
                 tp, (fixed_avg - 2*fixed_std/np.sqrt(trials))/bond_max, 'b:', linewidth=.5)
        plt.plot(tp, (var_avg + 2*var_std/np.sqrt(trials))/bond_max, 'b:',
                 tp, (var_avg - 2*var_std/np.sqrt(trials))/bond_max, 'b:', linewidth=.5)
        plt.plot(tp, pde_count, 'k', label='Deterministic')
        plt.legend(loc='best')
        plt.title('Bond numbers for each algorithm for a single receptor')
        plt.show()
