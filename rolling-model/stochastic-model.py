from constructA import *
import matplotlib.pyplot as plt

# Numerical Parameters
L, T = 2.0, 0.2
M, N, time_steps = 100, 100, 500
z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                              np.linspace(-np.pi/2, np.pi/2, N+1),
                              indexing='ij')
z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')
bond_list = np.array([])

t = np.linspace(0, T, time_steps+1)
h = z_mesh[1, 0] - z_mesh[0, 0]
nu = th_mesh[0, 1] - th_mesh[0, 0]
dt = t[1]-t[0]
lam = nu/h

# Biological Parameters
d_prime = .1
eta = .1
delta = 3.0
kap = 1.0
radius = 1.0
eta_v = .01
eta_om = .01

om_f = 20.0
v_f = (1 + d_prime)*om_f

# A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime)

om = np.zeros(time_steps + 1)
v = np.zeros(time_steps + 1)
om[0] = om_f
v[0] = v_f

# l_matrix = length(z_mesh, th_mesh, d_prime)

for i in range(time_steps):
    bond_list[:, 0] += -dt*v[i]  # Update z positions
    bond_list[:, 1] += -dt*om[i]  # Update theta positions

    probs = np.random.rand(bond_list.shape[0])
    break_indices = np.nonzero(probs < (1 - np.exp(
        -dt*np.exp(delta*length(bond_list[:, 0], bond_list[:, 1])))))


