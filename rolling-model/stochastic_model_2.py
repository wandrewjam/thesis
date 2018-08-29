from constructA import *
import matplotlib.pyplot as plt


def stochastic_model_tau(T=0.4, N=100, time_steps=1000, bond_max=100,
                         d_prime=0.1, eta=0.1, delta=3.0, kap=1.0, eta_v=0.01, eta_om=0.01, gamma=20.0):
    ####################################################################
    # This function runs a stochastic simulation with a fixed timestep #
    ####################################################################

    # Reduced model (without bond saturation)

    th_vec = np.linspace(-np.pi/2, np.pi/2, num=N+1)[:-1]
    nu = th_vec[1] - th_vec[0]
    th_vec += nu/2

    bond_list = np.empty(shape=(0, 2))  # Might need to make this a regular Python list

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    expected_vals = kap*np.sqrt(2*np.pi/eta)*np.exp(-eta/2*(1 - np.cos(th_vec) + d_prime)**2)*bond_max*dt

    om_f = gamma
    v_f = (1+d_prime)*gamma

    om = np.zeros(time_steps + 1)
    v = np.zeros(time_steps + 1)
    om[0] = om_f
    v[0] = v_f

    for i in range(time_steps):
        bond_list[:, 0] += -dt*v[i]  # Update z positions
        bond_list[:, 1] += -dt*om[i]  # Update theta positions
        # Reclassify theta bins
        bin_list = ((bond_list[:, 1] + np.pi/2)/nu).astype(dtype=int)  # I might not need this list
        break_indices = np.where(bin_list < 0)

        bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)

        # Decide which bonds break
        break_probs = np.random.rand(bond_list.shape[0])
        break_indices = np.append(arr=break_indices, values=np.nonzero(break_probs < (1 - np.exp(
            -dt*np.exp(delta*bond_lengths))))[0])

        # Generate bonds in each bin
        forming_bonds = np.random.poisson(lam=expected_vals)

        new_bonds = np.zeros(shape=(np.sum(forming_bonds), 2))
        counter = 0

        # Choose lengths, add to new bond array
        for j in np.where(forming_bonds)[0]:
            new_bonds[counter:counter+forming_bonds[j], 0] = np.random.normal(loc=np.sin(th_vec[j]),
                                                                              scale=np.sqrt(1/eta))
            new_bonds[counter:counter+forming_bonds[j], 1] = th_vec[j]
            counter += forming_bonds[j]

        # Update the bond array
        bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        # Calculate forces and torques
        zs = bond_list[:, 0]
        thetas = bond_list[:, 1]
        force = nu/bond_max*np.sum(a=zs-np.sin(thetas))  # Might need an additional factor of something
        torque = nu/bond_max*np.sum(a=(1-np.cos(thetas)+d_prime)*np.sin(thetas) +
                                      (np.sin(thetas)-zs)*np.cos(thetas))  # Might need an additional factor of something

        v[i+1], om[i+1] = v_f + force/eta_v, om_f + torque/eta_om
    return v, om, t


v, om, t = stochastic_model_tau()
plt.step(t, v)
plt.show()

plt.step(t, om)
plt.show()
