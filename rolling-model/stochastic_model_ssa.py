from constructA import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def stochastic_model_ssa(L=2.5, T=0.4, N=100, bond_max=100, d_prime=.1, eta=.1,
                         delta=3.0, kap=1.0, eta_v=.01, eta_om=.01):
    ########################################################################
    # This function runs a stochastic simulation using a variable timestep #
    ########################################################################
    
    th_vec = np.linspace(-np.pi/2, np.pi/2, num=N+1)[:-1]
    nu = th_vec[1] - th_vec[0]
    th_vec += nu/2

    bond_list = np.empty(shape=(0, 2))  # Might need to make this a regular Python list

    # Biological Parameters
    # d_prime = .1
    # eta = .1
    # delta = 3.0
    # kap = 1.0
    # eta_v = .01
    # eta_om = .01
    expected_coefs = kap*np.sqrt(2*np.pi/eta)*np.exp(-eta/2*(1 - np.cos(th_vec) + d_prime)**2)

    om_f = 20.0
    v_f = (1 + d_prime)*om_f

    # A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime)

    v, om = np.array([v_f]), np.array([om_f])
    n = np.array([0])
    t = np.array([0])
    i = 0

    start = timer()
    while t[-1] < T:
        # Reclassify theta bins
        bin_list = ((bond_list[:, 1] + np.pi/2)/nu).astype(dtype=int)  # I might not need this list
        break_indices = np.where(bin_list < 0)
        bin_list = bin_list[bin_list >= 0]  # I need to make sure bonds with attachments theta < -pi/2 break always

        bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)

        # Decide which bonds break
        break_rates = np.exp(delta*bond_lengths)

        # Decide which bonds form
        bond_counts = np.bincount(bin_list)  # I need this column to be dtype=int, maybe use a separate array
        bond_counts = np.append(bond_counts, values=np.zeros(shape=N-bond_counts.shape[0]))
        form_rates = expected_coefs*(bond_max - bond_counts)  # Calculate the expected values

        all_rates = np.append(break_rates, form_rates)
        sum_rates = np.cumsum(all_rates)
        a0 = sum_rates[-1]

        r = np.random.rand(2)
        dt = 1/a0*np.log(1/r[0])
        j = np.searchsorted(a=sum_rates, v=r[1]*a0)

        bond_list[:, 0] += -dt*v[-1]  # Update z positions
        bond_list[:, 1] += -dt*om[-1]  # Update theta positions
        t = np.append(t, t[-1]+dt)

        if j < break_rates.shape[0]:
            break_indices = np.append(break_indices, j)
        else:
            new_bonds = np.zeros(shape=(1, 2))
            new_bonds[0, 0] = np.random.normal(loc=np.sin(th_vec[j - break_rates.shape[0]]), scale=np.sqrt(1/eta))
            new_bonds[0, 1] = th_vec[j - break_rates.shape[0]]
            bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Update the bond array
        bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        n = np.append(arr=n, values=bond_list.shape[0])

        # Calculate forces and torques
        zs = bond_list[:, 0]
        thetas = bond_list[:, 1]
        force = nu/bond_max*np.sum(a=zs-np.sin(thetas))  # Might need an additional factor of something
        torque = nu/bond_max*np.sum(a=(1-np.cos(thetas)+d_prime)*np.sin(thetas) +
                                     (np.sin(thetas)-zs)*np.cos(thetas))  # Might need an additional factor of something

        v, om = np.append(arr=v, values=v_f + force/eta_v), np.append(arr=om, values=om_f + torque/eta_om)
    return v, om, t, n
