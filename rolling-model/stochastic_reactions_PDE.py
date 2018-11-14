import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import truncnorm
from constructA import length, nd_force, nd_torque
from timeit import default_timer as timer
from time import strftime


def fixed_motion(v, om, L=2.5, T=0.4, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1,
                 delta=3.0, kap=1.0, saturation=True, binding='both'):

    # Full Model
    th_vec = np.linspace(-np.pi, np.pi, num=2*N+1)[:-1]
    nu = th_vec[1] - th_vec[0]

    bond_list = np.empty(shape=(0, 2))

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    theta_arr = np.zeros(shape=(2*N, time_steps+1))
    theta_arr[:, 0] = th_vec

    def coeffs_and_bounds(bins):
        coeffs = kap*np.exp(-eta/2*(1 - np.cos(bins) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(bins) + L)) - erf(np.sqrt(eta/2)*(np.sin(bins) - L))
        )
        coeffs = (bins > -np.pi/2)*(bins < np.pi/2)*coeffs
        a = (-L - np.sin(bins))/np.sqrt(1/eta)
        b = (L - np.sin(bins))/np.sqrt(1/eta)
        return coeffs, a, b

    on = (binding == 'both') or (binding == 'on')
    off = (binding == 'both') or (binding == 'on')

    master_list = [bond_list]
    forces, torques = np.zeros(shape=time_steps+1), np.zeros(shape=time_steps+1)

    for i in range(time_steps):
        bond_list[:, 0] += -dt*v  # Update z positions
        th_vec += -dt*om  # Update theta bin positions
        th_vec = ((th_vec + np.pi) % (2*np.pi)) - np.pi  # Make sure bin positions are in [-pi, pi)
        theta_arr[:, i+1] = th_vec

        # Generate list of breaking indices
        break_indices = np.where(th_vec[bond_list[:, 1].astype(int)] < -np.pi/2)
        break_indices = np.append(break_indices, values=np.where(th_vec[bond_list[:, 1].astype(int)] > np.pi/2))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] > L))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] < -L))
        # break_indices = []

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


def variable_motion(v, om, L=2.5, T=0.4, N=100, bond_max=100, d_prime=.1, eta=.1,
                    delta=3.0, kap=1.0, saturation=True, binding='both'):

    th_vec = np.linspace(-np.pi, np.pi, num=2*N+1)[:-1]
    nu = th_vec[1] - th_vec[0]
    theta_list = [th_vec]

    bond_list = np.empty(shape=(0, 2))

    def coeffs_and_bounds(bins):
        coeffs = kap*np.exp(-eta/2*(1 - np.cos(bins) + d_prime)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(bins) + L)) - erf(np.sqrt(eta/2)*(np.sin(bins) - L))
        )
        coeffs = (bins > -np.pi/2)*(bins < np.pi/2)*coeffs
        a = (-L - np.sin(bins))/np.sqrt(1/eta)
        b = (L - np.sin(bins))/np.sqrt(1/eta)
        return coeffs, a, b

    on = (binding == 'both') or (binding == 'on')
    off = (binding == 'both') or (binding == 'on')

    master_list = [bond_list]
    t = np.array([0])
    forces, torques = np.array([0]), np.array([0])

    while t[-1] < T:
        # Generate list of breaking indices
        break_indices = np.where(th_vec[bond_list[:, 1].astype(int)] < -np.pi/2)
        break_indices = np.append(break_indices, values=np.where(th_vec[bond_list[:, 1].astype(int)] > np.pi/2))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] > L))
        break_indices = np.append(break_indices, values=np.where(bond_list[:, 0] < -L))

        # break_indices = []

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
        theta_list.append(th_vec)

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


def pde_motion(v, om, L=2.5, T=.4, M=100, N=100, time_steps=1000, d_prime=0.1, eta=0.1,
               delta=3.0, kap=1.0, saturation=True, binding='both'):
    # Numerical Parameters
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    dt = t[1]-t[0]

    on = (binding == 'both') or (binding == 'on')
    off = (binding == 'both') or (binding == 'on')

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


def pde_bins(v, om, L=2.5, T=.4, M=100, N=100, time_steps=1000, d_prime=0.1, eta=0.1,
             delta=3.0, kap=1.0, saturation=True, binding='both'):

    # Full Model
    z_vec = np.linspace(-L, L, 2*M+1)
    th_vec = np.linspace(-np.pi, np.pi, num=2*N+1)[:-1]
    h = z_vec[1] - z_vec[0]
    nu = th_vec[1] - th_vec[0]
    z_mesh, th_mesh = np.meshgrid(z_vec, th_vec, indexing='ij')

    t = np.linspace(0, T, time_steps+1)
    dt = t[1]-t[0]

    theta_arr = np.zeros(shape=(2*N, time_steps+1))
    theta_arr[:, 0] = th_vec

    on = (binding == 'both') or (binding == 'on')
    off = (binding == 'both') or (binding == 'on')

    m_mesh = np.zeros(shape=(2*M+1, 2*N, time_steps+1))  # Bond densities are initialized to zero
    forces, torques = np.zeros(shape=time_steps+1), np.zeros(shape=time_steps+1)
    l_new = length(z_mesh, th_mesh, d_prime)
    for i in range(time_steps):
        # bond_list[:, 0] += -dt*v  # Update z positions
        th_mesh += -dt*om  # Update theta bin positions
        th_mesh = ((th_mesh + np.pi) % (2*np.pi)) - np.pi  # Make sure bin positions are in [-pi, pi)
        theta_arr[:, i+1] = th_mesh[0, :]

        l_old = np.copy(l_new)
        l_new = length(z_mesh, th_mesh, d_prime)

        m_mesh[:-1, :, i+1] = (m_mesh[:-1, :, i] + v*dt/h*(m_mesh[1:, :, i] - m_mesh[:-1, :, i]) +
                               on*dt*kap*np.exp(-eta/2*l_old[:-1, :]**2) *
                               (1 - saturation*np.tile(np.trapz(m_mesh[:, :, i], z_vec, axis=0), reps=(2*M, 1)))) / \
                              (1 + dt*np.exp(delta*l_new[:-1, :]))
        m_mesh[:, :, i+1] = np.where((-np.pi/2 < th_mesh) * (th_mesh < np.pi/2), m_mesh[:, :, i+1], 0)

    return z_mesh, th_mesh, m_mesh, forces, torques


def count_fixed(v, om, L=2.5, T=0.4, N=100, time_steps=1000, bond_max=100, d_prime=0.1, eta=0.1, delta=3.0,
                kap=1.0, saturation=True, binding='both', **kwargs):
    start = timer()
    master_list, t, forces, torques = fixed_motion(v, om, L, T, N, time_steps, bond_max, d_prime, eta,
                                                                delta, kap, saturation, binding)
    count = [master_list[i].shape[0] for i in range(len(master_list))]
    end = timer()

    if 'k' in kwargs and 'trials' in kwargs:
        print('Completed {:d} of {:d} fixed runs. This run took {:g} seconds.'.format(kwargs['k']+1, kwargs['trials'],
                                                                                      end-start))
    elif 'k' in kwargs:
        print('Completed {:d} fixed runs so far. This run took {:g} seconds.'.format(kwargs['k']+1, end-start))
    elif 'trials' in kwargs:
        print('Completed one of {:d} fixed runs. This run took {:g} seconds.'.format(kwargs['trials'], end-start))
    else:
        print('Completed one fixed run. This run took {:g} seconds.'.format(end-start))

    return count, forces, torques


def count_variable(v, om, L=2.5, T=0.4, N=100, time_steps=1000, bond_max=100, d_prime=.1, eta=.1,
                   delta=3.0, kap=1.0, saturation=True, binding='both', **kwargs):
    start = timer()
    master_list, t, forces, torques = variable_motion(v, om, L, T, N, bond_max, d_prime, eta,
                                                      delta, kap, saturation, binding)
    t_sample = np.linspace(0, T, num=time_steps+1)
    count = [master_list[i].shape[0] for i in range(len(master_list))]
    end = timer()

    if 'k' in kwargs and 'trials' in kwargs:
        print('Completed {:d} of {:d} variable runs. This run took {:g} seconds.'.format(kwargs['k']+1, kwargs['trials'],
                                                                                      end-start))
    elif 'k' in kwargs:
        print('Completed {:d} variable runs so far. This run took {:g} seconds.'.format(kwargs['k']+1, end-start))
    elif 'trials' in kwargs:
        print('Completed one of {:d} variable runs. This run took {:g} seconds.'.format(kwargs['trials'], end-start))
    else:
        print('Completed one variable run. This run took {:g} seconds.'.format(end-start))

    indices = np.searchsorted(t, t_sample, side='right') - 1
    return np.array(count)[indices], np.array(forces)[indices], np.array(torques)[indices]


if __name__ == '__main__':
    trials = int(raw_input('Number of trials: '))
    fix_master_list = []
    var_master_list = []
    t_list = []

    # Parameters
    M = int(raw_input('M: '))
    N = int(raw_input('N: '))
    v = float(raw_input('v: '))
    om = float(raw_input('om: '))
    delta = 3
    T = 1
    init = None
    sat = True
    binding = 'both'
    time_steps = int(raw_input('time steps: '))
    bond_max = 10
    L = 2.5
    nu = np.pi/N

    proc = int(raw_input('Number of processes: '))

    f_forces, f_torques = np.zeros(shape=(trials, time_steps+1)), np.zeros(shape=(trials, time_steps+1))
    v_forces, v_torques = np.zeros(shape=(trials, time_steps+1)), np.zeros(shape=(trials, time_steps+1))

    z_mesh, th_mesh, m_mesh, tp, d_forces, d_torques = pde_motion(v, om, L=L, T=T, M=M, N=N, time_steps=time_steps,
                                                                  delta=delta, saturation=sat, binding=binding)
    m_bins = pde_bins(v, om, L=L, T=T, M=M, N=N, time_steps=time_steps, delta=delta, saturation=sat, binding=binding)[2]

    pde_count = np.trapz(np.trapz(m_mesh[:, :, :], z_mesh[:, 0], axis=0), th_mesh[0, :], axis=0)
    bin_count = np.sum(np.trapz(m_bins[:, :, :], z_mesh[:, 0], axis=0), axis=0)

    pool = mp.Pool(processes=proc)
    fixed_result = [pool.apply_async(count_fixed, args=(v, om),
                                      kwds={'L': L, 'T': T, 'N': N, 'time_steps': time_steps, 'bond_max': bond_max,
                                            'delta': delta, 'saturation': sat, 'binding': binding, 'k': k,
                                            'trials': trials}
                                      ) for k in range(trials)]
    var_result = [pool.apply_async(count_variable, args=(v, om),
                                    kwds={'L': L, 'T': T, 'N': N, 'time_steps': time_steps, 'bond_max': bond_max,
                                          'delta': delta, 'saturation': sat, 'binding': binding, 'k': k,
                                          'trials': trials}
                                    ) for k in range(trials)]

    fixed_forces = [f.get()[1] for f in fixed_result]  # Get the forces
    var_forces = [var.get()[1] for var in var_result]

    fixed_torques = [f.get()[2] for f in fixed_result]  # Get the torques
    var_torques = [var.get()[2] for var in var_result]

    fixed_result = [f.get()[0] for f in fixed_result]  # Get the bond counts
    var_result = [var.get()[0] for var in var_result]

    fixed_arr = np.vstack(fixed_result)
    var_arr = np.vstack(var_result)

    ffo_arr = np.vstack(fixed_forces)
    vfo_arr = np.vstack(var_forces)

    fto_arr = np.vstack(fixed_torques)
    vto_arr = np.vstack(var_torques)

    fixed_avg = np.mean(fixed_arr, axis=0)
    fixed_std = np.std(fixed_arr, axis=0)
    var_avg = np.mean(var_arr, axis=0)
    var_std = np.std(var_arr, axis=0)

    ffo_avg = np.mean(ffo_arr, axis=0)
    ffo_std = np.std(ffo_arr, axis=0)
    vfo_avg = np.mean(vfo_arr, axis=0)
    vfo_std = np.std(vfo_arr, axis=0)

    fto_avg = np.mean(fto_arr, axis=0)
    fto_std = np.std(fto_arr, axis=0)
    vto_avg = np.mean(vto_arr, axis=0)
    vto_std = np.std(vto_arr, axis=0)

    # Define parameter array and filename, and save the count data
    # The time is included to prevent overwriting an existing file
    par_array = np.array([delta, T, init, sat, binding, M, N, bond_max, L])
    file_path = './data/sta_rxns/'
    file_name = 'multimov_M{0:d}_N{1:d}_v{2:g}_om{3:g}_trials{4:d}_{5:s}.npz'.format(M, N, v, om, trials,
                                                                                     strftime('%d%m%y'))
    np.savez_compressed(file_path+file_name, par_array, fixed_arr, var_arr, pde_count, ffo_arr, fto_arr, vfo_arr,
                        vto_arr, bin_count, tp, par_array=par_array, fixed_array=fixed_arr, var_array=var_arr,
                        pde_count=pde_count, ffo_arr=ffo_arr, fto_arr=fto_arr, vfo_arr=vfo_arr, vto_arr=vto_arr,
                        bin_count=bin_count, tp=tp)
    print('Data saved in file {:s}'.format(file_name))

    plt.plot(tp[1:], (fixed_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed Step')
    plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'b:',
             tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'b:', linewidth=0.5)

    plt.plot(tp[1:], (var_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'g', label='Variable Step')
    plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'g:',
             tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max -
                      pde_count)[1:]/pde_count[1:], 'g:', linewidth=0.5)

    plt.plot(tp[1:], (bin_count*nu - pde_count)[1:]/pde_count[1:], 'r', label='PDE with bins')

    plt.plot(tp, np.zeros(shape=tp.shape), 'k')
    plt.legend()
    plt.title('Relative error of the stochastic simulation with fixed motion')
    plt.show()

    plt.plot(tp, fixed_avg*nu/bond_max, 'b', label='Fixed Step')
    plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max)[1:], 'b:',
             tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max)[1:], 'b:',
             linewidth=0.5)

    plt.plot(tp, var_avg*nu/bond_max, 'g', label='Variable Step')
    plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max)[1:], 'g:',
             tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max)[1:], 'g:',
             linewidth=0.5)

    plt.plot(tp, bin_count*nu, 'r', label='PDE with bins')
    plt.plot(tp, pde_count, 'k', label='PDE Solution')
    plt.legend()
    plt.title('Bond quantities of the stochastic simulations with fixed motion')
    plt.show()

    ind = 100
    plt.plot(tp[ind:], (ffo_avg - d_forces)[ind:]/d_forces[ind:], 'b', label='Fixed Step')
    plt.plot(tp[ind:], ((ffo_avg + 2*ffo_std/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'b:',
             tp[ind:], ((ffo_avg - 2*ffo_std/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'b:',
             linewidth=0.5)

    plt.plot(tp[ind:], (vfo_avg - d_forces)[ind:]/d_forces[ind:], 'g', label='Variable Step')
    plt.plot(tp[ind:], ((vfo_avg + 2*vfo_std/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'g:',
             tp[ind:], ((vfo_avg - 2*vfo_std/np.sqrt(trials)) - d_forces)[ind:]/d_forces[ind:], 'g:',
             linewidth=0.5)

    plt.plot(tp, np.zeros(shape=tp.shape), 'k')
    plt.legend()
    plt.title('Relative error of the forces of the stochastic simulation with fixed motion')
    plt.show()

    plt.plot(tp, ffo_avg, 'b', label='Fixed Step')
    plt.plot(tp, (ffo_avg + 2*ffo_std/np.sqrt(trials)), 'b:',
             tp, (ffo_avg - 2*ffo_std/np.sqrt(trials)), 'b:', linewidth=0.5)

    plt.plot(tp, vfo_avg, 'g', label='Variable Step')
    plt.plot(tp, (vfo_avg + 2*vfo_std/np.sqrt(trials)), 'g:',
             tp, (vfo_avg - 2*vfo_std/np.sqrt(trials)), 'g:', linewidth=0.5)

    plt.plot(tp, d_forces, 'k')
    plt.legend()
    plt.title('Relative error of the forces of the stochastic simulation with fixed motion')
    plt.show()

