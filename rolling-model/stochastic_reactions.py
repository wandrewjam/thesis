import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm
from constructA import length
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
    elif init is 'sat':
        bond_list = np.concatenate((np.random.uniform(low=-L, high=L, size=(bond_max*N, 1)),
                                    np.repeat(th_vec, bond_max)[:, None]), axis=1)
    else:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1
    master_list = [bond_list]

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    if ztype is 'cont_exact':
        # For continuous z, exact rate integral
        expected_coeffs = dt*kap*np.exp(-eta/2*(1 - np.cos(th_vec) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(th_vec) + z_vec[-1])) - erf(np.sqrt(eta/2)*(np.sin(th_vec) + z_vec[0]))
        )
        a = (z_vec[0] - th_vec)/np.sqrt(1/eta)
        b = (z_vec[-1] - th_vec)/np.sqrt(1/eta)
    elif ztype is 'cont_approx':
        # For continuous z, approximate rate integral
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        expected_coeffs = dt*kap*np.trapz(np.exp(-eta/2*l_matrix[1:-1, :]**2), z_vec[1:-1], axis=0)  # I'm not sure about this
        # expected_coeffs = dt*kap*h*np.sum(np.exp(-eta/2*l_matrix**2), axis=0)
        a = (z_vec[0] - th_vec)/np.sqrt(1/eta)
        b = (z_vec[-1] - th_vec)/np.sqrt(1/eta)
    elif ztype is 'discrete':
        # For discrete z
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        expected_coeffs = h*dt*kap*np.exp(-eta/2*l_matrix**2)

    for i in range(time_steps):

        if binding is 'both' or binding is 'off':
            # Decide which bonds break
            bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)
            break_probs = np.random.rand(bond_list.shape[0])
            break_indices = np.nonzero(break_probs < (1 - np.exp(
                -dt*np.exp(delta*bond_lengths))))[0]

        if binding is 'both' or binding is 'on':
            # Reclassify theta bins
            bin_list = ((bond_list[:, 1] + np.pi/2)/nu).astype(dtype=int)

            # Decide which bonds form
            bond_counts = np.bincount(bin_list)
            bond_counts = np.append(bond_counts, values=np.zeros(shape=N-bond_counts.shape[0]))
            if saturation:
                if ztype is 'cont_exact' or ztype is 'cont_approx':
                    # Continuous z formulation
                    expected_vals = expected_coeffs*(bond_max - bond_counts)  # Calculate the expected values
                elif ztype is 'discrete':
                    # Discrete z formulation
                    expected_vals = expected_coeffs*(bond_max - bond_counts[None, 1:])
            else:
                if ztype is 'cont_exact' or ztype is 'cont_approx':
                    # Continuous z formulation
                    expected_vals = expected_coeffs*bond_max  # Calculate the expected values without saturation
                elif ztype is 'discrete':
                    # Discrete z formulation
                    expected_vals = expected_coeffs*bond_max

            # Generate bonds in each bin
            if ztype is 'cont_exact' or ztype is 'cont_approx':
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
            elif ztype is 'discrete':
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
        if binding is 'both' or binding is 'off':
            bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        if binding is 'both' or binding is 'on':
            bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Store the bond list
        master_list.append(bond_list)
    return master_list, t


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
    elif init is 'sat':
        bond_list = np.concatenate((np.random.uniform(low=-L, high=L, size=(bond_max*N, 1)),
                                    np.repeat(th_vec, bond_max)[:, None]), axis=1)
    else:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1
    master_list = [bond_list]

    t = [0]
    if ztype is 'cont_exact':
        # For continuous z
        expected_coeffs = kap*np.exp(-eta/2*(1 - np.cos(th_vec) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(th_vec) + z_vec[-1])) - erf(np.sqrt(eta/2)*(np.sin(th_vec) + z_vec[0]))
        )
        a = (z_vec[0] - th_vec)/np.sqrt(1/eta)
        b = (z_vec[-1] - th_vec)/np.sqrt(1/eta)
    elif ztype is 'cont_approx':
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        # expected_coeffs = kap*np.trapz(np.exp(-eta/2*l_matrix**2), z_vec, axis=0)
        expected_coeffs = kap*np.trapz(np.exp(-eta/2*l_matrix[:, :]**2), z_vec, axis=0)
        a = (z_vec[0] - th_vec)/np.sqrt(1/eta)
        b = (z_vec[-2] - th_vec)/np.sqrt(1/eta)
    elif ztype is 'discrete':
        # For discrete z
        l_matrix = length(z_mesh, th_mesh, d_prime=d_prime)
        expected_coeffs = h*kap*np.exp(-eta/2*l_matrix**2)

    while t[-1] < T:
        break_rates = np.array([])
        form_rates = np.array([])
        if binding is 'both' or binding is 'off':
            # Decide which bonds break
            bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)
            break_rates = np.exp(delta*bond_lengths)

        if binding is 'both' or binding is 'on':
            # Reclassify theta bins
            bin_list = ((bond_list[:, 1] + np.pi/2)/nu).astype(dtype=int)  # I might not need this list
            #
            # bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)

            # Decide which bonds form
            bond_counts = np.bincount(bin_list)
            bond_counts = np.append(bond_counts, values=np.zeros(shape=N-bond_counts.shape[0]))
            if saturation:
                if ztype is 'cont_exact' or ztype is 'cont_approx':
                    # Continuous z formulation
                    form_rates = expected_coeffs*(bond_max - bond_counts)  # Calculate the expected values
                elif ztype is 'discrete':
                    # Discrete z formulation
                    form_rates = expected_coeffs*(bond_max - bond_counts[None, :])
            else:
                if ztype is 'cont_exact' or ztype is 'cont_approx':
                    # Continuous z formulation
                    form_rates = expected_coeffs*bond_max  # Calculate the expected values without saturation
                elif ztype is 'discrete':
                    # Discrete z formulation
                    form_rates = expected_coeffs*bond_max

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
                new_bonds = np.zeros(shape=(1, 2))
                new_bonds[0, 0] = truncnorm.rvs(a=a[index], b=b[index], loc=np.sin(th_vec[index]), scale=np.sqrt(1/eta))
                # new_bonds[0, 0] = np.random.normal(loc=np.sin(th_vec[j - break_rates.shape[0]]), scale=np.sqrt(1/eta))
                new_bonds[0, 1] = th_vec[index]
                bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)
        elif ztype is 'discrete':
            # Discrete z formulation
            all_rates = np.append(break_rates, form_rates.ravel(order='F'))
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
        master_list.append(bond_list)
    return master_list, t


def pde_reactions(init=None, L=2.5, T=.4, M=100, N=100, time_steps=1000, d_prime=0.1,
                  eta=0.1, delta=3.0, kap=1.0, saturation=True, binding='both'):

    # Numerical Parameters
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    dt = t[1]-t[0]

    m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))  # Bond densities are initialized to zero
    if init is 'sat':
        m_mesh[:, :, 0] = 0.2
    elif init is not None:
        # Handle a bad init input
        print('Error: Unknown initial distribution')
        return -1

    l_matrix = length(z_mesh, th_mesh, d_prime)

    form = binding is 'both' or binding is 'on'
    off = binding is 'both' or binding is 'off'
    if saturation:
        for i in range(time_steps):
            m_mesh[:, :, i+1] = (m_mesh[:, :, i] + form*dt*kap*np.exp(-eta/2*l_matrix**2) *
                                 (1 - np.tile(np.trapz(y=m_mesh[:, :, i], x=z_mesh[:, 0],
                                                       axis=0), reps=(2*M+1, 1)))) /\
                                 (1 + off*dt*np.exp(delta*l_matrix))
    else:
        for i in range(time_steps):
            m_mesh[:, :, i+1] = (m_mesh[:, :, i] + form*dt*kap*np.exp(-eta/2*l_matrix**2)) /\
                                 (1 + off*dt*np.exp(delta*l_matrix))

    return z_mesh, th_mesh, m_mesh, t


trials = 100
fix_master_list = []
var_master_list = []
t_list = []

## This code works for discrete z for both on-only cases. Except now I've changed it... and it doesn't work anymore
## Does this work for binding and unbinding as well? In the discrete case?
## It seems to work for unbinding alone, but not for binding and unbinding together? Check this tomorrow

## Maybe I need to make the theta vector midpoints of bins stretching from -pi/2 to pi/2
# Parameters
delta = 3
T = .4
init = None
sat = False
binding = 'on'
M, N = 128, 128
time_steps = 1000
bond_max = 10
L = 2.5
ztype = 'discrete'
nu = np.pi/N

for i in range(trials):
    start = timer()
    bond_list, ts = stochastic_reactions(init=init, L=L, T=T, M=M, N=N, bond_max=bond_max, time_steps=time_steps,
                                         delta=delta, saturation=sat, binding=binding, ztype=ztype)
    fix_master_list.append(bond_list)
    end = timer()
    print('{:d} of {:d} fixed time-step runs completed. This run took {:g} seconds.'.format(i+1, trials, end-start))

for i in range(trials):
    start = timer()
    bond_list, ts = ssa_reactions(init=init, L=L, T=T, M=M, N=N, bond_max=bond_max, delta=delta, saturation=sat,
                                  binding=binding, ztype=ztype)
    t_list.append(ts)
    # ts = ssa_reactions1(init=init, L=L, T=T, M=M, N=N, bond_max=bond_max, delta=delta, saturation=sat, binding=binding)
    # t_list1.append(ts)
    var_master_list.append(bond_list)
    end = timer()
    print('{:d} of {:d} variable time-step runs completed. This run took {:g} seconds.'.format(i+1, trials, end-start))
z_mesh, th_mesh, m_mesh, tp = pde_reactions(init=init, L=L, T=T, M=M, N=N, time_steps=time_steps, delta=delta,
                                            saturation=sat, binding=binding)

fix_sto_count = np.zeros(shape=(trials, tp.shape[0]))
var_sto_count = np.zeros(shape=(trials, tp.shape[0]))
# var_sto_count1 = np.zeros(shape=(trials, tp.shape[0]))

for i in range(trials):
    for j in range(len(fix_master_list[i])):
        fix_sto_count[i, j] = fix_master_list[i][j].shape[0]

for i in range(trials):
    temp_sto_count = np.zeros(shape=len(var_master_list[i]))
    for j in range(len(var_master_list[i])):
        temp_sto_count[j] = var_master_list[i][j].shape[0]
    var_sto_count[i, :] = temp_sto_count[np.searchsorted(t_list[i], tp, side='right')-1]

avg_fix_sto_count = np.mean(fix_sto_count, axis=0)
std_fix_sto_count = np.std(fix_sto_count, axis=0)
avg_var_sto_count = np.mean(var_sto_count, axis=0)
std_var_sto_count = np.std(var_sto_count, axis=0)

pde_count = np.trapz(np.trapz(m_mesh[:, :, :], z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)

# Define parameter array and filename, and save the count data
# The time is included to prevent overwriting an existing file
par_array = np.array([delta, T, init, sat, binding, M, N, bond_max, L, ztype])
file_path = './data/sta_rxns/'
file_name = 'M{0:d}_N{1:d}_ztype{2:s}_binding{3:s}_{4:s}.npz'.format(M, N, ztype, binding, strftime('%y%m%d-%H%M%S'))
np.savez(file_path+file_name, par_array, fix_sto_count, var_sto_count, pde_count, tp,
         par_array=par_array, fix_sto_count=fix_sto_count, var_sto_count=var_sto_count, pde_count=pde_count, tp=tp)
print('Data saved in file {:s}'.format(file_name))

plt.plot(tp[1:], (avg_fix_sto_count*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed Step')
plt.plot(tp[1:], ((avg_fix_sto_count + 2*std_fix_sto_count/np.sqrt(trials))*nu/bond_max -
                  pde_count)[1:]/pde_count[1:], 'b:',
         tp[1:], ((avg_fix_sto_count - 2*std_fix_sto_count/np.sqrt(trials))*nu/bond_max -
                  pde_count)[1:]/pde_count[1:], 'b:', linewidth=0.5)

plt.plot(tp[1:], (avg_var_sto_count*nu/bond_max - pde_count)[1:]/pde_count[1:], 'r', label='Variable Step')
plt.plot(tp[1:], ((avg_var_sto_count + 2*std_var_sto_count/np.sqrt(trials))*nu/bond_max -
                  pde_count)[1:]/pde_count[1:], 'r:',
         tp[1:], ((avg_var_sto_count - 2*std_var_sto_count/np.sqrt(trials))*nu/bond_max -
                  pde_count)[1:]/pde_count[1:], 'r:', linewidth=0.5)

plt.plot(tp, np.zeros(shape=tp.shape), 'k')
plt.legend()
if ztype is 'cont_exact' or ztype is 'cont_approx':
    plt.title('Relative error of the stochastic simulation with continuous z')
elif ztype is 'discrete':
    plt.title('Relative error of the stochastic simulation with discrete z')
plt.show()

plt.plot(tp, avg_fix_sto_count*nu/bond_max, 'b', label='Fixed Step')
plt.plot(tp[1:], ((avg_fix_sto_count + 2*std_fix_sto_count/np.sqrt(trials))*nu/bond_max)[1:], 'b:',
         tp[1:], ((avg_fix_sto_count - 2*std_fix_sto_count/np.sqrt(trials))*nu/bond_max)[1:], 'b:',
         linewidth=0.5)
plt.plot(tp, avg_var_sto_count*nu/bond_max, 'r', label='Variable Step')
plt.plot(tp[1:], ((avg_var_sto_count + 2*std_var_sto_count/np.sqrt(trials))*nu/bond_max)[1:], 'r:',
         tp[1:], ((avg_var_sto_count - 2*std_var_sto_count/np.sqrt(trials))*nu/bond_max)[1:], 'r:',
         linewidth=0.5)
plt.plot(tp, pde_count, 'k', label='PDE Solution')

plt.legend()
if ztype is 'cont_exact' or ztype is 'cont_approx':
    plt.title('Bond quantities of the stochastic simulations with continuous z')
elif ztype is 'discrete':
    plt.title('Bond quantities of the stochastic simulations with discrete z')
plt.show()
plt.show()
