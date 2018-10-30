import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import truncnorm
from constructA import length, nd_force, nd_torque
from timeit import default_timer as timer


def fixed_motion(L=2.5, T=0.4, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
                 delta=3.0, kap=1.0, gamma=20.0, saturation=True, binding='both'):

    # Full Model
    th_vec = np.linspace(-np.pi, np.pi, num=2*N+1)[:-1]
    nu = th_vec[1] - th_vec[0]

    bond_list = np.empty(shape=(0, 2))

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    def coeffs_and_bounds(bins):
        coeffs = kap*np.exp(-eta/2*(1 - np.cos(bins) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(bins) + L)) - erf(np.sqrt(eta/2)*(np.sin(bins) - L))
        )
        coeffs = (bins > -np.pi/2)*(bins < np.pi/2)*coeffs
        a = (-L - np.sin(bins))/np.sqrt(1/eta)
        b = (L - np.sin(bins))/np.sqrt(1/eta)
        return coeffs, a, b

    om = gamma
    v = (1 + d_prime)*gamma
    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'on')

    master_list = [bond_list]
    forces, torques = np.zeros(shape=time_steps+1), np.zeros(shape=time_steps+1)

    for i in range(time_steps):
        bond_list[:, 0] += -dt*v  # Update z positions
        th_vec += -dt*om  # Update theta bin positions
        th_vec = ((th_vec + np.pi) % (2*np.pi)) - np.pi  # Make sure bin positions are in [-pi, pi)

        # Generate list of breaking indices
        break_indices = np.where(th_vec[bond_list[:, 1].astype(int)] < -np.pi/2)
        break_indices = np.append(break_indices, values=np.where(th_vec[bond_list[:, 1].astype(int)] > np.pi/2))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] > L))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] < -L))

        bond_lengths = length(bond_list[:, 0], th_vec[bond_list[:, 1].astype(int)], d_prime=d_prime)

        # Decide which bonds break
        break_probs = np.random.rand(bond_list.shape[0])
        break_indices = np.append(arr=break_indices, values=np.nonzero(break_probs < (1 - np.exp(
            -off*dt*np.exp(delta*bond_lengths))))[0])

        # Decide which bonds form
        bond_counts = np.bincount(bond_list[:, 1].astype(int), minlength=2*N)
        expected_coeffs, a, b = coeffs_and_bounds(th_vec)

        expected_vals = on*dt*expected_coeffs*(bond_max - saturation*bond_counts)  # Calculate the expected values

        # Generate bonds in each bin
        expected_vals = expected_vals.clip(min=0)
        forming_bonds = np.random.poisson(lam=expected_vals)

        new_bonds = np.zeros(shape=(np.sum(forming_bonds), 2))
        counter = 0

        # Choose lengths, add to new bond array
        for j in np.where(forming_bonds)[0]:
            new_bonds[counter:counter+forming_bonds[j], 0] = truncnorm.rvs(a=a[j], b=b[j], loc=np.sin(th_vec[j]),
                                                                           scale=np.sqrt(1/eta),
                                                                           size=forming_bonds[j])
            new_bonds[counter:counter+forming_bonds[j], 1] = j
            counter += forming_bonds[j]

        # Update the bond array
        bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)
        bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)
        master_list.append(bond_list)

        # Calculate forces and torques
        zs = bond_list[:, 0]
        thetas = bond_list[:, 1].astype(int)
        forces[i+1] = nu/bond_max*np.sum(a=zs-np.sin(th_vec[thetas]))
        torques[i+1] = nu/bond_max*np.sum(a=(1-np.cos(th_vec[thetas])+d_prime)*np.sin(th_vec[thetas]) +
                                     (np.sin(th_vec[thetas])-zs)*np.cos(th_vec[thetas]))

    return master_list, t, forces, torques


def variable_motion(L=2.5, T=0.4, N=100, bond_max=100, d_prime=.1, eta=.1,
                    delta=3.0, kap=1.0, gamma=20, saturation=True, binding='both'):

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

    om = gamma
    v = (1 + d_prime)*gamma
    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'on')

    master_list = [bond_list]
    t = np.array([0])
    forces, torques = np.array([0]), np.array([0])

    while t[-1] < T:
        # Generate list of breaking indices
        break_indices = np.where(th_vec[bond_list[:, 1].astype(int)] < -np.pi/2)
        break_indices = np.append(break_indices, values=np.where(th_vec[bond_list[:, 1].astype(int)] > np.pi/2))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] > L))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] < -L))

        bond_lengths = length(bond_list[:, 0], th_vec[bond_list[:, 1].astype(int)], d_prime=d_prime)

        # Decide which bonds break
        break_rates = off*np.exp(delta*bond_lengths)

        # Decide which bonds form
        bond_counts = np.bincount(bond_list[:, 1].astype(int), minlength=2*N)
        expected_coeffs, a, b = coeffs_and_bounds(th_vec)
        form_rates = on*expected_coeffs*(bond_max - saturation*bond_counts)  # Calculate the expected values

        all_rates = np.append(break_rates, form_rates)
        sum_rates = np.cumsum(all_rates)
        a0 = sum_rates[-1]

        r = np.random.rand(2)
        dt = 1/a0*np.log(1/r[0])
        j = np.searchsorted(a=sum_rates, v=r[1]*a0)

        bond_list[:, 0] += -dt*v  # Update z positions
        th_vec += -dt*om  # Update theta bin positions
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
        forces = np.append(forces, nu/bond_max*np.sum(a=zs-np.sin(th_vec[thetas])))
        torques = np.append(torques, nu/bond_max*np.sum(a=(1-np.cos(th_vec[thetas])+d_prime)*np.sin(th_vec[thetas]) +
                                                          (np.sin(th_vec[thetas])-zs)*np.cos(th_vec[thetas])))

    return master_list, t, forces, torques


def pde_motion(L=2.5, T=.4, M=100, N=100, time_steps=1000, d_prime=0.1, eta=0.1,
               delta=3.0, kap=1.0, gamma=20.0, saturation=True, binding='both'):
    # Numerical Parameters
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    dt = t[1]-t[0]

    om = gamma
    v = (1 + d_prime)*gamma
    on = (binding is 'both') or (binding is 'on')
    off = (binding is 'both') or (binding is 'on')

    m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))  # Bond densities are initialized to zero
    l_matrix = length(z_mesh, th_mesh, d_prime)

    forces, torques = np.zeros(shape=time_steps+1), np.zeros(shape=time_steps+1)

    # Check for the CFL condition
    if om*dt/nu > 1:
        print('Warning: the CFL condition for theta is not satisfied!')
    if v*dt/h > 1:
        print('Warning: the CFL condition for z is not satisfied!')

    for i in range(time_steps):
        m_mesh[:-1, :-1, i+1] = (m_mesh[:-1, :-1, i] + om*dt/nu*(m_mesh[:-1, 1:, i] - m_mesh[:-1, :-1, i]) +
                                 v*dt/h*(m_mesh[1:, :-1, i] - m_mesh[:-1, :-1, i]) +
                                 on*dt*kap*np.exp(-eta/2*l_matrix[:-1, :-1]**2) *
                                 (1 - saturation*np.tile(np.trapz(m_mesh[:, :-1, i], z_mesh[:, 0], axis=0), reps=(2*M, 1)))) /\
                                  (1 + off*dt*np.exp(delta*l_matrix[:-1, :-1]))

        forces[i+1] = nd_force(m_mesh[:, :, i+1], z_mesh, th_mesh)  # Changed force calculations to integrate over all z and th meshes
        torques[i+1] = nd_torque(m_mesh[:, :, i+1], z_mesh, th_mesh, d_prime)

    return z_mesh, th_mesh, m_mesh, t, forces, torques


trials = 100
fix_master_list = []
var_master_list = []
t_list = []

# Parameters
gamma = 18
delta = 3
T = 0.4
init = None
sat = True
binding = 'both'
M, N = 128, 128
time_steps = 1000
bond_max = 10
L = 2.5
nu = np.pi/N

f_forces, f_torques = np.zeros(shape=(trials, time_steps+1)), np.zeros(shape=(trials, time_steps+1))
v_forces, v_torques = np.zeros(shape=(trials, time_steps+1)), np.zeros(shape=(trials, time_steps+1))

z_mesh, th_mesh, m_mesh, tp, d_forces, d_torques = pde_motion(L=L, T=T, M=M, N=N, time_steps=time_steps, delta=delta,
                                                              gamma=gamma, saturation=sat, binding=binding)

for i in range(trials):
    start = timer()
    bond_list, ts, f_forces[i, :], f_torques[i, :] = fixed_motion(L=L, T=T, N=N, time_steps=time_steps,
                                                                  bond_max=bond_max, delta=delta, gamma=gamma,
                                                                  saturation=sat, binding=binding)
    fix_master_list.append(bond_list)
    end = timer()
    print('{:d} of {:d} fixed time-step runs completed. This run took {:g} seconds.'.format(i+1, trials, end-start))

for i in range(trials):
    start = timer()
    bond_list, ts, forces, torques = variable_motion(L=L, T=T, N=N, bond_max=bond_max, delta=delta, gamma=gamma,
                                                     saturation=sat, binding=binding)
    t_list.append(ts)
    # ts = ssa_reactions1(init=init, L=L, T=T, M=M, N=N, bond_max=bond_max, delta=delta, saturation=sat, binding=binding)
    # t_list1.append(ts)
    v_forces[i, :] = forces[np.searchsorted(ts, tp, side='right')-1]
    v_torques[i, :] = torques[np.searchsorted(ts, tp, side='right')-1]
    var_master_list.append(bond_list)

    end = timer()
    print('{:d} of {:d} variable time-step runs completed. This run took {:g} seconds.'.format(i+1, trials, end-start))

fix_sto_count = np.zeros(shape=(trials, tp.shape[0]))
var_sto_count = np.zeros(shape=(trials, tp.shape[0]))
var_sto_count1 = np.zeros(shape=(trials, tp.shape[0]))

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

avg_fix_sto_force = np.mean(f_forces, axis=0)
std_fix_sto_force = np.std(f_forces, axis=0)
avg_var_sto_force = np.mean(v_forces, axis=0)
std_var_sto_force = np.std(v_forces, axis=0)

avg_fix_sto_torqu = np.mean(f_torques, axis=0)
std_fix_sto_torqu = np.std(f_torques, axis=0)
avg_var_sto_torqu = np.mean(v_torques, axis=0)
std_var_sto_torqu = np.std(v_torques, axis=0)

pde_count = np.trapz(np.trapz(m_mesh[:, :, :], z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)

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
plt.title('Relative error of the stochastic simulation with fixed motion')
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
plt.title('Bond quantities of the stochastic simulations with fixed motion')
plt.show()

ind = 100
plt.plot(tp[ind:], (avg_fix_sto_force - d_forces)[ind:]/d_forces[ind:], 'b', label='Fixed Step')
plt.plot(tp[ind:], ((avg_fix_sto_force + 2*std_fix_sto_force/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'b:',
         tp[ind:], ((avg_fix_sto_force - 2*std_fix_sto_force/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'b:',
         linewidth=0.5)

plt.plot(tp[ind:], (avg_var_sto_force - d_forces)[ind:]/d_forces[ind:], 'r', label='Variable Step')
plt.plot(tp[ind:], ((avg_var_sto_force + 2*std_var_sto_force/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'r:',
         tp[ind:], ((avg_var_sto_force - 2*std_var_sto_force/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'r:',
         linewidth=0.5)

plt.plot(tp, np.zeros(shape=tp.shape), 'k')
plt.legend()
plt.title('Relative error of the forces of the stochastic simulation with fixed motion')
plt.show()

plt.plot(tp, avg_fix_sto_force, 'b', label='Fixed Step')
plt.plot(tp, (avg_fix_sto_force + 2*std_fix_sto_force/np.sqrt(trials)), 'b:',
         tp, (avg_fix_sto_force - 2*std_fix_sto_force/np.sqrt(trials)), 'b:', linewidth=0.5)

plt.plot(tp, avg_var_sto_force, 'r', label='Variable Step')
plt.plot(tp, (avg_var_sto_force + 2*std_var_sto_force/np.sqrt(trials)), 'r:',
         tp, (avg_var_sto_force - 2*std_var_sto_force/np.sqrt(trials)), 'r:', linewidth=0.5)

plt.plot(tp, d_forces, 'k')
plt.legend()
plt.title('Relative error of the forces of the stochastic simulation with fixed motion')
plt.show()

