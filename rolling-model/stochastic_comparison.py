from stochastic_model import *
from stochastic_model_ssa import *
from sampling import *
from time_dependent import time_dependent
import matplotlib.pyplot as plt
from timeit import default_timer as timer

##################### Check the forces! Compare with the known solutions #############################


def stochastic_comparison(num_iterations=100, num_samples=200, M=128, N=128,
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

    v3_sampled = np.zeros(shape=(num_iterations, num_samples))
    t_sampled = np.linspace(start=0, stop=0.4, num=num_samples)

    # v1_array, om1_array = np.zeros(shape=(num_iterations, time_steps+1)), \
    #                       np.zeros(shape=(num_iterations, time_steps+1))
    # v2_array, om2_array = np.zeros(shape=(num_iterations, time_steps+1)), \
    #                       np.zeros(shape=(num_iterations, time_steps+1))
    v3_list, om3_list, t3_list = [], [], []
    # n1_array = np.zeros(shape=(num_iterations, time_steps+1))
    # n2_array = np.zeros(shape=(num_iterations, time_steps+1))
    n3_list = list()

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

    for i in range(num_iterations):
        start = timer()
        v3, om3, bond_list_temp, t3 = stochastic_model_ssa(
            N=N, bond_max=bond_max, eta_v=eta_v, eta_om=eta_om, gamma=gamma)
        v3_list.append(v3)
        om3_list.append(om3)
        n_temp = np.zeros(shape=len(bond_list_temp))
        for j in range(len(bond_list_temp)):
            n_temp[j] = bond_list_temp[j].shape[0]
        n3_list.append(n_temp)
        t3_list.append(t3)

        v3_sampled[i, :] = sample_velocity(t3, v3, t_samp=t_sampled)
        end = timer()
        print('Completed {:d} of {:d} variable time-step experiments. '
              'This run took {:g} seconds.'.format(
               i+1, num_iterations, end - start))

    np.savez_compressed('./data/v_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *v3_list)
    np.savez_compressed('./data/om_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *om3_list)
    np.savez_compressed('./data/t_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *t3_list)
    np.savez_compressed('./data/n_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *n3_list)

    v_avg = np.mean(v3_sampled, axis=0)
    v_std = np.std(v3_sampled, axis=0)

    if plot:
        plt.plot(t_sampled, v_avg, 'k',
                 t_sampled[1:], (v_avg + v_std)[1:], 'r--',
                 t_sampled[1:], (v_avg - v_std)[1:], 'r--')
        plt.show()
    return None


if __name__ == '__main__':
    exp = int(raw_input('exponent: '))
    eta_v = float(raw_input('eta_v and eta_om: '))
    stochastic_comparison(M=2**exp, N=2**exp, time_steps=10*2**exp,
                          bond_max=10, gamma=20.0, num_iterations=5,
                          eta_v=eta_v, eta_om=eta_v)
    print('Done!')
