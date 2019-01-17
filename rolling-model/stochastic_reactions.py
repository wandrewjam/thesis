import multiprocessing as mp
import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm
from utils import length
from timeit import default_timer as timer
from time import strftime
import matplotlib.pyplot as plt


def stochastic_reactions(init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
                         delta=3.0, kap=1.0, saturation=True, binding='both', ztype='cont_exact'):

    z_vec = np.linspace(-L, L, 2*M+1)[:-1]
    th_vec = np.linspace(-np.pi/2, np.pi/2, N+1)[:-1]
    z_mesh, th_mesh = np.meshgrid(z_vec, th_vec, indexing='ij')
    h = z_vec[1] - z_vec[0]
    nu = th_vec[1] - th_vec[0]
    th_vec += nu/2
    z_vec += h/2

    if init is None:
        bond_list = np.empty(shape=(0, 2))
    elif init == 'sat':
        bond_list = np.concatenate((np.random.uniform(low=-L, high=L, size=(bond_max*N, 1)),
                                    np.repeat(th_vec, bond_max)[:, None]), axis=1)
    else:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1
    bond_numbers = [bond_list.shape[0]]

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    if ztype == 'cont_exact':
        # For continuous z, exact rate integral
        expected_coeffs = dt*kap*np.exp(-eta/2*(1 - np.cos(th_vec) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(th_vec) + L)) - erf(np.sqrt(eta/2)*(np.sin(th_vec) - L))
        )
        a = (-L - np.sin(th_vec))/np.sqrt(1/eta)
        b = (L - np.sin(th_vec))/np.sqrt(1/eta)
    elif ztype == 'cont_approx':
        # For continuous z, approximate rate integral
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        expected_coeffs = dt*kap*np.trapz(np.exp(-eta/2*l_matrix**2), z_vec, axis=0)  # I'm not sure about this
        # expected_coeffs = dt*kap*h*np.sum(np.exp(-eta/2*l_matrix**2), axis=0)
        a = (-L - th_vec)/np.sqrt(1/eta)
        b = (L - th_vec)/np.sqrt(1/eta)
    elif ztype == 'discrete':
        # For discrete z
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        expected_coeffs = h*dt*kap*np.exp(-eta/2*l_matrix**2)

    for i in range(time_steps):

        if binding == 'both' or binding == 'off':
            # Decide which bonds break
            bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)
            break_probs = np.random.rand(bond_list.shape[0])
            break_indices = np.nonzero(break_probs < (1 - np.exp(
                -dt*np.exp(delta*bond_lengths))))[0]

        if binding == 'both' or binding == 'on':
            # Reclassify theta bins
            bin_list = ((bond_list[:, 1] + np.pi/2)/nu).astype(dtype=int)

            # Decide which bonds form
            bond_counts = np.bincount(bin_list)
            bond_counts = np.append(bond_counts, values=np.zeros(shape=N-bond_counts.shape[0]))
            if saturation:
                if ztype == 'cont_exact' or ztype == 'cont_approx':
                    # Continuous z formulation
                    expected_vals = expected_coeffs*(bond_max - bond_counts)  # Calculate the expected values
                elif ztype == 'discrete':
                    # Discrete z formulation
                    expected_vals = expected_coeffs*(bond_max - bond_counts[None, :])
            else:
                if ztype == 'cont_exact' or ztype == 'cont_approx':
                    # Continuous z formulation
                    expected_vals = expected_coeffs*bond_max  # Calculate the expected values without saturation
                elif ztype == 'discrete':
                    # Discrete z formulation
                    expected_vals = expected_coeffs*bond_max

            # Generate bonds in each bin
            if ztype == 'cont_exact' or ztype == 'cont_approx':
                # Continuous z formulation
                expected_vals = expected_vals.clip(min=0)
                forming_bonds = np.random.poisson(lam=expected_vals)

                new_bonds = np.zeros(shape=(np.sum(forming_bonds), 2))
                counter = 0

                # Choose lengths, add to new bond array
                for j in np.where(forming_bonds)[0]:
                    # new_bonds[counter:counter+forming_bonds[j], 0] = np.random.normal(loc=np.sin(th_vec[j]),
                    #                                                                   scale=np.sqrt(1/eta),
                    #                                                                   size=forming_bonds[j])
                    new_bonds[counter:counter+forming_bonds[j], 0] = truncnorm.rvs(a=a[j], b=b[j], loc=np.sin(th_vec[j]),
                                                                                   scale=np.sqrt(1/eta),
                                                                                   size=forming_bonds[j])
                    new_bonds[counter:counter+forming_bonds[j], 1] = th_vec[j]
                    counter += forming_bonds[j]
            elif ztype == 'discrete':
                # Discrete z formulation
                expected_vals = expected_vals.clip(min=0)
                forming_bonds = np.random.poisson(lam=expected_vals)

                nonzeros = np.transpose(forming_bonds.nonzero())
                new_bonds = np.zeros(shape=nonzeros.shape)
                counter = 0

                for j, k in nonzeros:
                    new_bonds[counter:counter+forming_bonds[j, k]] = [[z_vec[j], th_vec[k]]]
                    counter += forming_bonds[j, k]

        # Update the bond array
        if binding == 'both' or binding == 'off':
            bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        if binding == 'both' or binding == 'on':
            bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Store the bond list
        bond_numbers.append(bond_list.shape[0])
    return bond_numbers, t


def ssa_reactions(init=None, L=2.5, T=0.4, M=100, N=100, bond_max=100, d_prime=0.1, eta=0.1,
                  delta=3.0, kap=1.0, saturation=True, binding='both', ztype='cont_exact'):

    z_vec = np.linspace(-L, L, 2*M+1)[:-1]
    th_vec = np.linspace(-np.pi/2, np.pi/2, N+1)[:-1]
    z_mesh, th_mesh = np.meshgrid(z_vec, th_vec, indexing='ij')
    h = z_vec[1] - z_vec[0]
    nu = th_vec[1] - th_vec[0]
    th_vec += nu/2
    z_vec += h/2

    if init is None:
        bond_list = np.empty(shape=(0, 2))
    elif init == 'sat':
        bond_list = np.concatenate((np.random.uniform(low=-L, high=L, size=(bond_max*N, 1)),
                                    np.repeat(th_vec, bond_max)[:, None]), axis=1)
    else:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1
    bond_numbers = [bond_list.shape[0]]

    t = [0]
    if ztype == 'cont_exact':
        # For continuous z
        expected_coeffs = kap*np.exp(-eta/2*(1 - np.cos(th_vec) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(th_vec) + L)) - erf(np.sqrt(eta/2)*(np.sin(th_vec) - L))
        )
        a = (-L - np.sin(th_vec))/np.sqrt(1/eta)
        b = (L - np.sin(th_vec))/np.sqrt(1/eta)
    elif ztype == 'cont_approx':
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        # expected_coeffs = kap*np.trapz(np.exp(-eta/2*l_matrix**2), z_vec, axis=0)
        expected_coeffs = kap*np.trapz(np.exp(-eta/2*l_matrix[:, :]**2), z_vec, axis=0)
        a = (-L - th_vec)/np.sqrt(1/eta)
        b = (L - th_vec)/np.sqrt(1/eta)
    elif ztype == 'discrete':
        # For discrete z
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        expected_coeffs = h*kap*np.exp(-eta/2*l_matrix**2)

    while t[-1] < T:
        break_rates = np.array([])
        form_rates = np.array([])
        if binding == 'both' or binding == 'off':
            # Decide which bonds break
            bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)
            break_rates = np.exp(delta*bond_lengths)

        if binding == 'both' or binding == 'on':
            # Reclassify theta bins
            bin_list = ((bond_list[:, 1] + np.pi/2)/nu).astype(dtype=int)  # I might not need this list
            #
            # bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)

            # Decide which bonds form
            bond_counts = np.bincount(bin_list)
            bond_counts = np.append(bond_counts, values=np.zeros(shape=N-bond_counts.shape[0]))
            if saturation:
                if ztype == 'cont_exact' or ztype == 'cont_approx':
                    # Continuous z formulation
                    form_rates = expected_coeffs*(bond_max - bond_counts)  # Calculate the expected values
                elif ztype == 'discrete':
                    # Discrete z formulation
                    form_rates = expected_coeffs*(bond_max - bond_counts[None, :])
            else:
                if ztype == 'cont_exact' or ztype == 'cont_approx':
                    # Continuous z formulation
                    form_rates = expected_coeffs*bond_max  # Calculate the expected values without saturation
                elif ztype == 'discrete':
                    # Discrete z formulation
                    form_rates = expected_coeffs*bond_max

        # Generate bonds in each bin
        if ztype == 'cont_exact' or ztype == 'cont_approx':
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
                new_bonds = np.zeros(shape=(1, 2))
                new_bonds[0, 0] = truncnorm.rvs(a=a[index], b=b[index], loc=np.sin(th_vec[index]), scale=np.sqrt(1/eta))
                # new_bonds[0, 0] = np.random.normal(loc=np.sin(th_vec[j - break_rates.shape[0]]), scale=np.sqrt(1/eta))
                new_bonds[0, 1] = th_vec[index]
                bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)
        elif ztype == 'discrete':
            # Discrete z formulation
            all_rates = np.append(break_rates, form_rates.ravel(order='F'))
            if all_rates.shape[0] == 0:
                a0 = 1e-10
            else:
                sum_rates = np.cumsum(all_rates)
                a0 = sum_rates[-1]

            r = np.random.rand(2)
            dt = 1/a0*np.log(1/r[0])
            j = np.searchsorted(a=sum_rates, v=r[1]*a0)
            t = np.append(t, t[-1] + dt)

            if j < break_rates.shape[0]:
                bond_list = np.delete(arr=bond_list, obj=j, axis=0)
            else:
                index = j - break_rates.shape[0]
                new_bonds = np.zeros(shape=(1, 2))
                new_bonds[0, 0] = z_mesh.ravel(order='F')[index]
                new_bonds[0, 1] = th_mesh.ravel(order='F')[index]
                bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Store the bond list
        bond_numbers.append(bond_list.shape[0])
    return bond_numbers, t


def pde_reactions(init=None, L=2.5, T=.4, M=100, N=100, time_steps=1000, d_prime=0.1,
                  eta=0.1, delta=3.0, kap=1.0, saturation=True, binding='both'):

    # Numerical Parameters
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    m_mesh = np.zeros(shape=(2*M+1, N+1))  # Bond densities are initialized to zero
    bond_numbers = np.zeros(shape=time_steps+1)

    if init == 'sat':
        m_mesh = 0.2
        bond_numbers[0] = np.trapz(np.trapz(m_mesh, z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)
    elif init is not None:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1

    l_matrix = length(z_mesh, th_mesh, d_prime)

    form = binding == 'both' or binding == 'on'
    off = binding == 'both' or binding == 'off'
    if saturation:
        for i in range(time_steps):
            m_mesh = (m_mesh + form*dt*kap*np.exp(-eta/2*l_matrix**2) * (1 - np.tile(
                np.trapz(y=m_mesh, x=z_mesh[:, 0], axis=0), reps=(2*M+1, 1)))) / \
                     (1 + off*dt*np.exp(delta*l_matrix))
            bond_numbers[i+1] = np.trapz(np.trapz(m_mesh, z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)
    else:
        for i in range(time_steps):
            m_mesh = (m_mesh + form*dt*kap*np.exp(-eta/2*l_matrix**2)) / (1 + off*dt*np.exp(delta*l_matrix))
            bond_numbers[i+1] = np.trapz(np.trapz(m_mesh, z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)

    return z_mesh, th_mesh, bond_numbers, t


def count_fixed(init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, bond_max=100, d_prime=0.1,
                eta=0.1, delta=3.0, kap=1.0, saturation=True, binding='both', ztype='cont_exact', **kwargs):
    start = timer()
    count, t = stochastic_reactions(init, L, T, M, N, time_steps, bond_max, d_prime, eta, delta,
                                    kap, saturation, binding, ztype)
    end = timer()

    if 'k' in kwargs and 'trials' in kwargs:
        print('Completed {:d} of {:d} fixed runs. This run took {:g} seconds.'.format(kwargs['k']+1, kwargs['trials'],
                                                                                      end-start))
    elif 'k' in kwargs:
        print('Completed {:d} fixed runs so far. This run took {:g} seconds.'.format(kwargs['k']+1, end-start))
    elif 'trials' in kwargs:
        print('Completed one of {:d} fixed runs. This run took {:g} seconds.'.format(kwargs['trials'], end-start))
    else:
        print('Completed one fixed run. This run took {:g} seconds.'.format(end-start))

    return np.array(count)


def count_variable(init=None, L=2.5, T=0.4, M=100, N=100, time_steps=1000, bond_max=100, d_prime=0.1,
                   eta=0.1, delta=3.0, kap=1.0, saturation=True, binding='both', ztype='cont_exact', **kwargs):
    start = timer()
    count, t = ssa_reactions(init, L, T, M, N, bond_max, d_prime, eta,
                             delta, kap, saturation, binding, ztype)
    t_sample = np.linspace(0, T, num=time_steps+1)
    end = timer()

    if 'k' in kwargs and 'trials' in kwargs:
        print('Completed {:d} of {:d} variable runs. This run took {:g} seconds.'.format(kwargs['k']+1, kwargs['trials'],
                                                                                      end-start))
    elif 'k' in kwargs:
        print('Completed {:d} variable runs so far. This run took {:g} seconds.'.format(kwargs['k']+1, end-start))
    elif 'trials' in kwargs:
        print('Completed one of {:d} variable runs. This run took {:g} seconds.'.format(kwargs['trials'], end-start))
    else:
        print('Completed one variable run. This run took {:g} seconds.'.format(end-start))

    return np.array(count)[np.searchsorted(t, t_sample, side='right')-1]


if __name__ == '__main__':
    trials = int(input('Number of trials: '))

    # With a few runs, this code seems to work for the saturating, formation and breaking case with continuous z

    # This seems to work for the variable time-step algo for on/off with saturation, but not the fixed time-step.
    # (Both discrete and continuous z cases)

    # Parameters
    delta = 3
    T = 1
    init = None
    sat = True
    binding = 'both'
    M = int(input('M: '))
    N = int(input('N: '))
    time_steps = int(input('time steps: '))
    bond_max = 10
    L = 2.5
    ztype = 'cont_exact'
    nu = np.pi/N

    pool = mp.Pool(processes=4)
    fixed_result = [pool.apply_async(count_fixed, kwds={'init': init, 'T': T, 'M': M, 'N': N, 'time_steps': time_steps,
                                                        'bond_max': bond_max, 'delta': delta, 'saturation': sat,
                                                        'binding': binding, 'ztype': ztype, 'k': k, 'trials': trials})
                    for k in range(trials)]

    var_result = [pool.apply_async(count_variable, kwds={'init': init, 'T': T, 'M': M, 'N': N, 'time_steps': time_steps,
                                                         'bond_max': bond_max, 'delta': delta, 'saturation': sat,
                                                         'binding': binding, 'ztype': ztype, 'k': k, 'trials': trials})
                  for k in range(trials)]

    fixed_result = [f.get() for f in fixed_result]
    var_result = [v.get() for v in var_result]

    fixed_arr = np.vstack(fixed_result)
    var_arr = np.vstack(var_result)

    fixed_avg = np.mean(fixed_arr, axis=0)
    var_avg = np.mean(var_arr, axis=0)
    fixed_std = np.std(fixed_arr, axis=0)
    var_std = np.std(var_arr, axis=0)

    z_mesh, th_mesh, pde_count, tp = pde_reactions(init=init, L=L, T=T, M=M, N=N, time_steps=time_steps, delta=delta,
                                                   saturation=sat, binding=binding)

    # Define parameter array and filename, and save the count data
    # The time is included to prevent overwriting an existing file
    par_array = np.array([delta, T, init, sat, binding, M, N, bond_max, L, ztype])
    file_path = './data/sta_rxns/'
    file_name = 'multirec_M{0:d}_N{1:g}_trials{2:d}_ztype{3:s}_{4:s}.npz'.format(M, N, trials, ztype,
                                                                                 strftime('%d%m%y'))
    np.savez_compressed(file_path+file_name, par_array, fixed_arr, var_arr, pde_count, tp, par_array=par_array,
                        fixed_array=fixed_arr, var_array=var_arr, pde_count=pde_count, tp=tp)
    print('Data saved in file {:s}'.format(file_name))

    plt.plot(tp[1:], (fixed_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed Step')
    plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'b:',
             tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'b:', linewidth=0.5)

    plt.plot(tp[1:], (var_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'r', label='Variable Step')
    plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'r:',
             tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'r:', linewidth=0.5)

    plt.plot(tp, np.zeros(shape=tp.shape), 'k')
    plt.legend()
    if ztype == 'cont_exact' or ztype == 'cont_approx':
        plt.title('Relative error of the stochastic simulation with continuous z')
    elif ztype == 'discrete':
        plt.title('Relative error of the stochastic simulation with discrete z')
    plt.show()

    plt.plot(tp, fixed_avg*nu/bond_max, 'b', label='Fixed Step')
    plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max)[1:], 'b:',
             tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max)[1:], 'b:',
             linewidth=0.5)
    plt.plot(tp, var_avg*nu/bond_max, 'r', label='Variable Step')
    plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max)[1:], 'r:',
             tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max)[1:], 'r:',
             linewidth=0.5)
    plt.plot(tp, pde_count, 'k', label='PDE Solution')

    plt.legend()
    if ztype == 'cont_exact' or ztype == 'cont_approx':
        plt.title('Bond quantities of the stochastic simulations with continuous z')
    elif ztype == 'discrete':
        plt.title('Bond quantities of the stochastic simulations with discrete z')
    plt.show()
    plt.show()
