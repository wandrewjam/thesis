from stochastic_model import *
from stochastic_model_ssa import *
from sampling import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from guppy import hpy
from memory_profiler import profile

##################### Check the forces! Compare with the known solutions #############################


@profile
def stochastic_comparison(num_iterations=100, num_samples=200, M=100, N=100,
                          time_steps=1000, bond_max=100, kap=1.0, delta=3.0,
                          eta_v=.01, eta_om=.01, gamma=20.0, plot=True):
    v3_sampled = np.zeros(shape=(num_iterations, num_samples))
    t_sampled = np.linspace(start=0, stop=0.4, num=num_samples)

    v1_array, om1_array = np.zeros(shape=(num_iterations, time_steps+1)), \
                          np.zeros(shape=(num_iterations, time_steps+1))
    v2_array, om2_array = np.zeros(shape=(num_iterations, time_steps+1)), \
                          np.zeros(shape=(num_iterations, time_steps+1))
    v3_list, om3_list, t3_list = [], [], []
    n1_array = np.zeros(shape=(num_iterations, time_steps+1))
    n2_array, n3_list = np.zeros(shape=(num_iterations, time_steps+1)), []

    for i in range(num_iterations):
        start = timer()
        v1_array[i, :], om1_array[i, :], bond_list_temp, t1 = \
            stochastic_model(N=N, time_steps=time_steps, bond_max=bond_max,
                             eta_v=eta_v, eta_om=eta_om, gamma=gamma)
        for j in range(time_steps+1):
            n1_array[i, j] = bond_list_temp[j].shape[0]
        v2_array[i, :], om2_array[i, :], bond_list_temp, t2 = \
            stochastic_model(N=N, time_steps=time_steps, bond_max=bond_max,
                             eta_v=eta_v, eta_om=eta_om, gamma=gamma,
                             saturation=False)
        for j in range(time_steps+1):
            n2_array[i, j] = bond_list_temp[j].shape[0]
        end = timer()
        print('Completed {:d} of {:d} fixed time-step experiments. '
              'This run took {:g} seconds.'.format(i+1, num_iterations,
                                                   end - start))

    print('\n')

    np.savez('./data/alg1_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), t1, v1_array, om1_array, t1=t1,
        v1=v1_array, om1=om1_array, n1=n1_array)
    np.savez('./data/alg2_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), t2, v2_array, om2_array, t2=t2,
        v2=v2_array, om2=om2_array, n2=n2_array)

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

    np.savez('./data/v_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *v3_list)
    np.savez('./data/om_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *om3_list)
    np.savez('./data/t_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
        N, bond_max, eta_om, gamma), *t3_list)
    np.savez('./data/n_ssa_N{:d}_bmax{:d}_eta{:g}_gamma{:g}'.format(
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
    exp = 8
    stochastic_comparison(M=2**exp, N=2**exp, time_steps=2000, bond_max=10,
                          gamma=20.0, num_iterations=1)
    print('Done!')
    h = hpy()
    print h.heap()
