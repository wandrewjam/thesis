from utils import *
# from scipy.sparse import eye
import matplotlib.pyplot as plt

# Parameters
L, T = 2.0, .1
M, N, time_steps = 100, 100, 200
z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                              np.linspace(-np.pi/2, np.pi/2, N+1),
                              indexing='ij')
# z_vec, th_vec = np.ravel(z_mesh, order='F'), np.ravel(th_mesh, order='F')


m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))

h = z_mesh[1, 0] - z_mesh[0, 0]
nu = th_mesh[0, 1] - th_mesh[0, 0]
dt = T/time_steps
lam = nu/h

d_prime = .1
eta = .1
delta = 3.0
kap = 1.0
radius = 1.0
eta_v = .01
eta_om = .01

om_f = 0.0
v_f = (1 + d_prime)*om_f

l_matrix = length(z_mesh, th_mesh, d_prime)

# Check for the CFL condition

if om_f*dt/nu > 1:
    print('Warning: the CFL condition for theta is not satisfied!')
if v_f*dt/h > 1:
    print('Warning: the CFL condition for z is not satisfied!')

# A, B, C, D, R = construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime)

for i in range(time_steps):
    m_mesh[:-1, :-1, i+1] = (m_mesh[:-1, :-1, i] + om_f*dt/nu*(m_mesh[:-1, 1:, i] - m_mesh[:-1, :-1, i]) +
                             v_f*dt/h*(m_mesh[1:, :-1, i] - m_mesh[:-1, :-1, i]) +
                             dt*kap*np.exp(-eta/2*l_matrix[:-1, :-1]**2) *
                             (1 - h*np.tile(np.sum(m_mesh[:-1, :-1, i], axis=0), reps=(2*M, 1)))) /\
                            (1 + dt*np.exp(delta*l_matrix[:-1, :-1]))

for i in range(11):
    plt.pcolormesh(z_mesh, th_mesh, m_mesh[:, :, 20*i])
    plt.colorbar()
    plt.plot(np.sin(np.linspace(-np.pi/2, np.pi/2, num=N+1)), np.linspace(-np.pi/2, np.pi/2, num=N+1), 'k-')
    plt.show()
