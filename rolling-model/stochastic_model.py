from utils import *
from scipy.special import erf
from scipy.stats import truncnorm
from timeit import default_timer as timer


def stochastic_model(L=2.5, T=0.4, N=100, time_steps=1000, bond_max=100,
                     d_prime=0.1, eta=0.1, delta=3.0, kap=1.0, eta_v=0.01,
                     eta_om=0.01, gamma=20.0, saturation=True):
    #####################################################################
    # This function runs a stochastic simulation using a fixed timestep #
    #####################################################################

    # Full Model
    th_vec = np.linspace(-np.pi, np.pi, num=2*N+1)[:-1]
    nu = th_vec[1] - th_vec[0]

    bond_list = np.empty(shape=(0, 2))

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    # Restrict the z values to those allowed in the PDE
    def coeffs_and_bounds(bins):
        coeffs = kap*np.exp(-eta/2*(1 - np.cos(bins) + d_prime)**2) *\
                 np.sqrt(np.pi/(2*eta)) *\
                 (erf(np.sqrt(eta/2)*(np.sin(bins) + L)) -
                  erf(np.sqrt(eta/2)*(np.sin(bins) - L)))

        coeffs = (bins > -np.pi/2)*(bins < np.pi/2)*coeffs
        a = (-L - np.sin(bins))/np.sqrt(1/eta)
        b = (L - np.sin(bins))/np.sqrt(1/eta)
        return coeffs, a, b

    om_f = gamma
    v_f = (1 + d_prime)*gamma

    om = np.zeros(time_steps + 1)
    v = np.zeros(time_steps + 1)
    master_list = [bond_list]
    om[0] = om_f
    v[0] = v_f

    # l_matrix = length(z_mesh, th_mesh, d_prime)
    start = timer()
    for i in range(time_steps):
        bond_list[:, 0] += -dt*v[i]  # Update z positions
        th_vec += -dt*om[i]  # Update theta bin positions
        # Make sure bin positions are in [-pi, pi)
        th_vec = ((th_vec + np.pi) % (2*np.pi)) - np.pi

        # Generate list of breaking indices
        break_indices = np.where(th_vec[bond_list[:, 1].astype(int)] <
                                 -np.pi/2)
        break_indices = np.append(break_indices, values=np.where(
            th_vec[bond_list[:, 1].astype(int)] > np.pi/2))
        break_indices = np.append(break_indices, values=np.where(
            bond_list[:, 0] > L))
        break_indices = np.append(break_indices, values=np.where(
            bond_list[:, 0] < -L))

        bond_lengths = length(bond_list[:, 0],
                              th_vec[bond_list[:, 1].astype(int)],
                              d_prime=d_prime)

        # Decide which bonds break
        break_probs = np.random.rand(bond_list.shape[0])
        break_indices = np.append(arr=break_indices,
                                  values=np.nonzero(break_probs < (1 - np.exp(
                                      -dt*np.exp(delta*bond_lengths))))[0])

        # Decide which bonds form
        bond_counts = np.bincount(bond_list[:, 1].astype(int), minlength=2*N)
        expected_coeffs, a, b = coeffs_and_bounds(th_vec)
        if saturation:
            # Calculate the expected values
            expected_vals = dt*expected_coeffs*(bond_max - bond_counts)
        else:
            # Calculate the expected values without saturation
            expected_vals = dt*expected_coeffs*bond_max

        # Generate bonds in each bin
        expected_vals = expected_vals.clip(min=0)
        forming_bonds = np.random.poisson(lam=expected_vals)

        new_bonds = np.zeros(shape=(np.sum(forming_bonds), 2))
        counter = 0

        # Choose lengths, add to new bond array
        for j in np.where(forming_bonds)[0]:
            new_bonds[counter:counter+forming_bonds[j], 0] = \
                truncnorm.rvs(a=a[j], b=b[j], loc=np.sin(th_vec[j]),
                              scale=np.sqrt(1/eta), size=forming_bonds[j])
            new_bonds[counter:counter+forming_bonds[j], 1] = j
            counter += forming_bonds[j]

        # Update the bond array
        bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)
        master_list.append(bond_list)

        # Calculate forces and torques
        zs = bond_list[:, 0]
        thetas = bond_list[:, 1].astype(int)
        force = nu/bond_max*np.sum(a=zs-np.sin(th_vec[thetas]))
        torque = nu/bond_max*np.sum(
            a=(1-np.cos(th_vec[thetas])+d_prime) * np.sin(th_vec[thetas]) +
              (np.sin(th_vec[thetas])-zs)*np.cos(th_vec[thetas])
        )

        v[i+1], om[i+1] = v_f + force/eta_v, om_f + torque/eta_om
    return v, om, master_list, t

