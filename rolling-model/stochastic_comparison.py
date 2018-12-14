import multiprocessing as mp
from stochastic_model import *
from stochastic_model_ssa import *
from sampling import *
from time_dependent import time_dependent
import matplotlib.pyplot as plt
from timeit import default_timer as timer


##################### Check the forces! Compare with the known solutions #############################


def count_variable(gamma, L=2.5, T=0.4, N=100, bond_max=100,
                   d_prime=.1, eta=.1, delta=3.0, kap=1.0, eta_v=.01,
                   eta_om=.01, **kwargs):
    start = timer()
    v, om, master_list, t = (
        stochastic_model_ssa(L=L, T=T, N=N, bond_max=bond_max, d_prime=d_prime,
                             eta=eta, delta=delta, kap=kap, eta_v=eta_v,
                             eta_om=eta_om, gamma=gamma)
    )

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

    return v, om, count, t


def stochastic_comparison(trials=100, num_samples=200, M=128, N=128,
                          time_steps=1000, bond_max=100, kap=1.0, delta=3.0,
                          eta_v=.01, eta_om=.01, gamma=20.0, plot=True):
    bw_v, bw_om, bw_mesh, bw_t = (
        time_dependent(M=M, N=N, time_steps=time_steps, eta_om=eta_om,
                       eta_v=eta_om, gamma=gamma, save_m=True, scheme='bw')
    )

    up_v, up_om, up_mesh, up_t = (
        time_dependent(M=M, N=N, time_steps=time_steps, eta_om=eta_om,
                       eta_v=eta_om, gamma=gamma, save_m=True, scheme='up')
    )

    np.savez_compressed('./data/bw_M{:d}_N{:d}_eta{:g}_gamma{:g}'.format(
        M, N, eta_om, gamma), bw_v, bw_om, bw_mesh, bw_t, bw_v=bw_v,
        bw_om=bw_om, bw_mesh=bw_mesh, bw_t=bw_t)
    np.savez_compressed('./data/up_M{:d}_N{:d}_eta{:g}_gamma{:g}'.format(
        M, N, eta_om, gamma), up_v, up_om, up_mesh, up_t, up_v=up_v,
        up_om=up_om, up_mesh=up_mesh, up_t=up_t)

    # v1_array, om1_array = np.zeros(shape=(num_iterations, time_steps+1)), \
    #                       np.zeros(shape=(num_iterations, time_steps+1))
    # v2_array, om2_array = np.zeros(shape=(num_iterations, time_steps+1)), \
    #                       np.zeros(shape=(num_iterations, time_steps+1))

    # n1_array = np.zeros(shape=(num_iterations, time_steps+1))
    # n2_array = np.zeros(shape=(num_iterations, time_steps+1))

    # for i in range(num_iterations):
    #     start = timer()
    #     v1_array[i, :], om1_array[i, :], bond_list_temp, t1 = \
    #         stochastic_model(N=N, time_steps=time_steps, bond_max=bond_max,
    #                          eta_v=eta_v, eta_om=eta_om, gamma=gamma)
    #     for j in range(time_steps+1):
    #         n1_array[i, j] = bond_list_temp[j].shape[0]
    #     v2_array[i, :], om2_array[i, :], bond_list_temp, t2 = \
    #         stochastic_model(N=N, time_steps=time_steps, bond_max=bond_max,
    #                          eta_v=eta_v, eta_om=eta_om, gamma=gamma,
    #                          saturation=False)
    #     for j in range(time_steps+1):
    #         n2_array[i, j] = bond_list_temp[j].shape[0]
    #     end = timer()
    #     print('Completed {:d} of {:d} fixed time-step experiments. '
    #           'This run took {:g} seconds.'.format(i+1, num_iterations,
    #                                                end - start))
    #
    # print('\n')

    # np.savez_compressed('./data/alg1_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
    #     N, bond_max, eta_om, gamma), t1, v1_array, om1_array, t1=t1,
    #     v1=v1_array, om1=om1_array, n1=n1_array)
    # np.savez_compressed('./data/alg2_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
    #     N, bond_max, eta_om, gamma), t2, v2_array, om2_array, t2=t2,
    #     v2=v2_array, om2=om2_array, n2=n2_array)

    # for i in range(num_iterations):
    #     start = timer()
    #     v3, om3, bond_list_temp, t3 = stochastic_model_ssa(
    #         N=N, bond_max=bond_max, eta_v=eta_v, eta_om=eta_om, gamma=gamma)
    #     v3_list.append(v3)
    #     om3_list.append(om3)
    #     n_temp = np.zeros(shape=len(bond_list_temp))
    #     for j in range(len(bond_list_temp)):
    #         n_temp[j] = bond_list_temp[j].shape[0]
    #     n3_list.append(n_temp)
    #     t3_list.append(t3)
    #
    #     v3_sampled[i, :] = sample_velocity(t3, v3, t_samp=t_sampled)
    #     end = timer()
    #     print('Completed {:d} of {:d} variable time-step experiments. '
    #           'This run took {:g} seconds.'.format(
    #            i+1, num_iterations, end - start))

    proc = int(raw_input('Processes: '))
    pool = mp.Pool(processes=proc)
    var_result = [
        pool.apply_async(count_variable, args=(gamma,),
                         kwds={'N': N, 'time_steps': time_steps,
                               'bond_max': bond_max, 'kap': kap, 'delta': delta,
                               'eta_v': eta_v, 'eta_om': eta_om, 'k': k,
                               'trials': trials}
                         ) for k in range(trials)
    ]

    v_list = [var.get()[0] for var in var_result]
    om_list = [var.get()[1] for var in var_result]
    n_list = [var.get()[2] for var in var_result]
    t_list = [var.get()[3] for var in var_result]

    np.savez_compressed('./data/v_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *v_list)
    np.savez_compressed('./data/om_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *om_list)
    np.savez_compressed('./data/t_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *t_list)
    np.savez_compressed('./data/n_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *n_list)

    # v_avg = np.mean(v3_sampled, axis=0)
    # v_std = np.std(v3_sampled, axis=0)
    #
    # if plot:
    #     plt.plot(t_sampled, v_avg, 'k',
    #              t_sampled[1:], (v_avg + v_std)[1:], 'r--',
    #              t_sampled[1:], (v_avg - v_std)[1:], 'r--')
    #     plt.show()
    return None


if __name__ == '__main__':
    exp = int(raw_input('exponent: '))
    trials = int(raw_input('trials: '))
    eta_v = float(raw_input('eta_v and eta_om: '))
    stochastic_comparison(M=2**exp, N=2**exp, time_steps=10*2**exp,
                          bond_max=10, gamma=20.0, trials=trials,
                          eta_v=eta_v, eta_om=eta_v)
    print('Done!')
