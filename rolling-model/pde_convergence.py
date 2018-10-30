import numpy as np
import matplotlib.pyplot as plt


######### Code for evaluating convergence of the full nonlinear model #########
L, T = 2.5, 0.4
eta_v = 0.01
gamma = 20
top = 8
bottom = 2
v = list()
om = list()
m = list()
t = list()
err = np.zeros(shape=top-bottom)
errm = np.zeros(shape=top-bottom)
errv = np.zeros(shape=top-bottom)
errom = np.zeros(shape=top-bottom)

# Look separately at errs in v and omega and m

for j in range(bottom, top):
    M = 2**j
    N = 2**j
    time_steps = 25*2**j
    file_str = './data/PDEdata_M{:d}_N{:d}_tsteps{:d}_eta{:g}_gamma{:g}_1order.npz'.format(
            M, N, time_steps, eta_v, gamma)

    data = np.load(file_str)
    v.append(data['v'])
    om.append(data['om'])
    m.append(data['m_mesh'])
    t.append(data['t'])
    print('Opened level {:d}'.format(j))

# Open the "exact" solution
M = 2**top
N = 2**top
time_steps = 25*2**top

file_str = './data/PDEdata_M{:d}_N{:d}_tsteps{:d}_eta{:g}_gamma{:g}_1order.npz'.format(
        M, N, time_steps, eta_v, gamma)
data_fine = np.load(file_str)
v_fine, om_fine, m_fine, t_fine = data_fine['v'], data_fine['om'], data_fine['m_mesh'], data_fine['t']
print('Opened level {:d}'.format(top))

for j in range(bottom, top):
    # Compute L2 error for each time step
    l2_m = L * np.pi * 2**(-2*j) * np.sum((m[j-bottom] -
                                           m_fine[::2**(top-j), ::2**(top-j),
                                           ::2**(top-j)])**2, axis=(0, 1))
    l2_v, l2_om = (v[j-bottom]-v_fine[::2**(top-j)])**2, (om[j-bottom] - om_fine[::2**(top-j)])**2

    # Compute L2 error across time steps
    err[j-bottom] = np.sqrt(T/(25*2**j)*np.sum(l2_m + l2_v + l2_om))
    errm[j-bottom] = np.sqrt(T/(25*2**j)*np.sum(l2_m))
    errv[j-bottom] = np.sqrt(T/(25*2**j)*np.sum(l2_v))
    errom[j-bottom] = np.sqrt(T/(25*2**j)*np.sum(l2_om))

plt.loglog(L/(2**np.arange(bottom, top)), err, 'o-', label='$L^2$ error')
# plt.loglog(L/(2**np.arange(bottom, top)), errm, 'o-', label='$m$ error')
# plt.loglog(L/(2**np.arange(bottom, top)), errv, 'o-', label='$v$ error')
# plt.loglog(L/(2**np.arange(bottom, top)), errom, 'o-', label='$\\omega$ error')
plt.loglog(L/(2**np.arange(bottom, top)), 7*L/(2**np.arange(bottom, top)), 'k:', label='Reference line')
plt.title('Convergence of errors for the full PDE model')
plt.xlabel('$h$')
plt.xscale('log', basex=2)
plt.ylabel('Error')
plt.legend()
plt.show()

for j in range(bottom, top):
    plt.plot(t[j - bottom], v[j - bottom], label='$h = {:.3g}$'.format(L/2**j))
plt.plot(t_fine, v_fine, label='$h = {:.3g}$'.format(L/2**top))
plt.title('Velocities for different discretization levels')
plt.xlabel('Time')
plt.ylabel('Linear velocity')
plt.legend()
plt.show()

for j in range(bottom, top):
    plt.plot(t[j - bottom], L*np.pi*2**(-2*j)*np.sum(m[j-bottom], axis=(0, 1)), label='$h = {:.3g}$'.format(L/2**j))
plt.plot(t_fine, L*np.pi*2**(-2*top)*np.sum(m_fine, axis=(0, 1)), label='$h = {:.3g}$'.format(L/2**top))
plt.title('Bond quantity for different discretization levels')
plt.xlabel('Time')
plt.ylabel('Bond quantity')
plt.legend()
plt.show()

# Print out table of convergence

title_str = '  M | time steps | L^2 error | Convergence '
table_str = ' {:2d} | {:10d} | {:9g} | {:11g} '

print(title_str)
print('-'*43)
for j in range(bottom, top):
    if j == bottom:
        print(table_str.format(2**j, 25*2**j, err[j - bottom], 0))
    else:
        print(table_str.format(2**j, 25*2**j, err[j - bottom], -np.log2(err[j - bottom]/err[j - 1 - bottom])))

