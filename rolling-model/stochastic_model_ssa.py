from constructA import *
from scipy.special import erf
from scipy.stats import truncnorm
from timeit import default_timer as timer


def stochastic_model_ssa(L=2.5, T=0.4, N=100, bond_max=100, d_prime=.1, eta=.1,
                         delta=3.0, kap=1.0, eta_v=.01, eta_om=.01, gamma=20):
    ########################################################################
    # This function runs a stochastic simulation using a variable timestep #
    ########################################################################
    
    th_vec = np.linspace(-np.pi, np.pi, num=2*N+1)[:-1]
    nu = th_vec[1] - th_vec[0]

    bond_list = np.empty(shape=(0, 2))

    def coeffs_and_bounds(bins):
        coeffs = kap*np.exp(-eta/2*(1 - np.cos(bins) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(bins) + L)) - erf(np.sqrt(eta/2)*(np.sin(bins) - L))
        )
        coeffs = (bins > -np.pi/2)*(bins < np.pi/2)*coeffs
        a = (-L - np.sin(bins))/np.sqrt(1/eta)
        b = (L - np.sin(bins))/np.sqrt(1/eta)
        return coeffs, a, b

    om_f = gamma
    v_f = (1 + d_prime)*gamma

    v, om = np.array([v_f]), np.array([om_f])
    master_list = [bond_list]
    t = np.array([0])

    start = timer()
    while t[-1] < T:
        # Generate list of breaking indices
        break_indices = np.where(th_vec[bond_list[:, 1].astype(int)] < -np.pi/2)
        break_indices = np.append(break_indices, values=np.where(th_vec[bond_list[:, 1].astype(int)] > np.pi/2))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] > L))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] < -L))

        bond_lengths = length(bond_list[:, 0], th_vec[bond_list[:, 1].astype(int)], d_prime=d_prime)

        # Decide which bonds break
        break_rates = np.exp(delta*bond_lengths)

        # Decide which bonds form
        bond_counts = np.bincount(bond_list[:, 1].astype(int), minlength=2*N)
        expected_coeffs, a, b = coeffs_and_bounds(th_vec)
        form_rates = expected_coeffs*(bond_max - bond_counts)  # Calculate the expected values

        all_rates = np.append(break_rates, form_rates)
        sum_rates = np.cumsum(all_rates)
        a0 = sum_rates[-1]

        r = np.random.rand(2)
        dt = 1/a0*np.log(1/r[0])
        j = np.searchsorted(a=sum_rates, v=r[1]*a0)

        bond_list[:, 0] += -dt*v[-1]  # Update z positions
        th_vec += -dt*om[-1]  # Update theta bin positions
        th_vec = ((th_vec + np.pi) % (2*np.pi)) - np.pi  # Make sure bin positions are in [-pi, pi)

        t = np.append(t, t[-1]+dt)

        if j < break_rates.shape[0]:
            break_indices = np.append(break_indices, j)
        else:
            index = j - break_rates.shape[0]
            new_bonds = np.zeros(shape=(1, 2))
            new_bonds[0, 0] = truncnorm.rvs(a=a[index], b=b[index], loc=np.sin(th_vec[index]), scale=np.sqrt(1/eta))
            new_bonds[0, 1] = index
            bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Update the bond array
        bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        master_list.append(bond_list)

        # Calculate forces and torques
        zs = bond_list[:, 0]
        thetas = bond_list[:, 1].astype(int)
        force = nu/bond_max*np.sum(a=zs-np.sin(th_vec[thetas]))
        torque = nu/bond_max*np.sum(a=(1-np.cos(th_vec[thetas])+d_prime)*np.sin(th_vec[thetas]) +
                                     (np.sin(th_vec[thetas])-zs)*np.cos(th_vec[thetas]))

        v, om = np.append(arr=v, values=v_f + force/eta_v), np.append(arr=om, values=om_f + torque/eta_om)
    return v, om, master_list, t
