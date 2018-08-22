from constructA import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Numerical Parameters
L, T = 2.0, 0.4
M, N, time_steps = 100, 100, 1000
bond_max = 1000

# Maybe don't need these?
# z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
#                               np.linspace(-np.pi/2, np.pi/2, N+1),
#                               indexing='ij')
# z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')

th_vec = np.linspace(-np.pi/2, np.pi/2, num=N+1)[:-1]
nu = th_vec[1] - th_vec[0]
th_vec += nu/2

bond_list = np.empty(shape=(0, 2))  # Might need to make this a regular Python list

t = np.linspace(0, T, time_steps+1)
# h = z_mesh[1, 0] - z_mesh[0, 0]
# nu = th_mesh[0, 1] - th_mesh[0, 0]
dt = t[1]-t[0]
# lam = nu/h

# Biological Parameters
d_prime = .1
eta = .1
delta = 3.0
kap = 1.0
radius = 1.0
eta_v = .01
eta_om = .01
expected_coefs = kap*np.sqrt(2*np.pi/eta)*np.exp(-eta/2*(1 - np.cos(th_vec) + d_prime)**2)

if np.max(expected_coefs)*dt*bond_max > 0.5:
    print('Warning! There is a large probability that more than 1 bond forms in a time step')

om_f = 20.0
v_f = (1 + d_prime)*om_f

# A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime)

om = np.zeros(time_steps + 1)
v = np.zeros(time_steps + 1)
om[0] = om_f
v[0] = v_f

# l_matrix = length(z_mesh, th_mesh, d_prime)
start = timer()
for i in range(time_steps):
    bond_list[:, 0] += -dt*v[i]  # Update z positions
    bond_list[:, 1] += -dt*om[i]  # Update theta positions
    # Reclassify theta bins
    bin_list = ((bond_list[:, 1] + np.pi/2)/nu).astype(dtype=int)  # I might not need this list
    break_indices = np.where(bin_list < 0)
    bin_list = bin_list[bin_list >= 0]  # I need to make sure bonds with attachments theta < -pi/2 break always

    bond_lengths = length(bond_list[:, 0], bond_list[:, 1], d_prime=d_prime)

    # Decide which bonds break
    break_probs = np.random.rand(bond_list.shape[0])
    break_indices = np.append(arr=break_indices, values=np.nonzero(break_probs < (1 - np.exp(
        -dt*np.exp(delta*bond_lengths))))[0])
    # break_indices = np.nonzero(break_probs < (1 - np.exp(-dt*np.exp(delta*bond_lengths))))[0]

    # Decide which bonds form
    bond_counts = np.bincount(bin_list)  # I need this column to be dtype=int, maybe use a separate array
    bond_counts = np.append(bond_counts, values=np.zeros(shape=N-bond_counts.shape[0]))
    expected_vals = expected_coefs*(bond_max - bond_counts)*dt  # Calculate the expected values

    # Generate bonds in each bin
    forming_probs = np.random.rand(N)
    forming_bonds = forming_probs > np.exp(-expected_vals)

    new_bonds = np.zeros(shape=(np.sum(forming_bonds), 2))
    counter = 0

    # Choose lengths, add to new bond array
    for j in np.where(forming_bonds)[0]:
        # new_length = 1 - np.cos(th_vec[j]) + d_prime + np.abs(
        #     np.random.normal(loc=0, scale=np.sqrt(1/eta)))

        new_bonds[counter:counter+forming_bonds[j], 0] = np.random.normal(loc=np.sin(th_vec[j]),
                                                                          scale=np.sqrt(1/eta))

        new_bonds[counter:counter+1, 1] = th_vec[j]
        counter += 1

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

end = timer()
print(end-start)
# plt.step(t, np.cumsum(v)/(np.arange(v.shape[0])+1))
plt.step(t, v)
plt.show()

# plt.step(t, np.cumsum(om)/(np.arange(om.shape[0])+1))
plt.step(t, om)
plt.show()

# How do I verify these results?
