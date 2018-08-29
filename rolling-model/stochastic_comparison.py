from stochastic_model import *
from stochastic_model_2 import *
from stochastic_model_ssa import *
from sampling import *


def stochastic_comparison(num_iterations=100, num_samples=200, bond_max=100, plot=True):
    v3_sampled = np.zeros(shape=(num_iterations, num_samples))
    t_sampled = np.linspace(start=0, stop=0.4, num=num_samples)

    v1_array, om1_array = np.zeros(shape=(num_iterations, 1001)), np.zeros(shape=(num_iterations, 1001))
    v2_array, om2_array = np.zeros(shape=(num_iterations, 1001)), np.zeros(shape=(num_iterations, 1001))
    v3_list, om3_list, t3_list = [], [], []

    for i in range(num_iterations):
        v1_array[i, :], om2_array[i, :], t1 = stochastic_model(bond_max=bond_max)
        v2_array[i, :], om2_array[i, :], t2 = stochastic_model_tau(bond_max=bond_max)
        print('Completed %i of %i fixed time-step experiments' % (i+1, num_iterations))

    print('\n')

    np.savez('alg1_bmax{:d}'.format(bond_max), t1, v1_array, om1_array, t1=t1, v1=v1_array, om1=om1_array)
    np.savez('alg2_bmax{:d}'.format(bond_max), t2, v2_array, om2_array, t2=t2, v2=v2_array, om2=om2_array)

    for i in range(num_iterations):
        v3, om3, t3, n = stochastic_model_ssa(bond_max=bond_max)
        v3_list.append(v3)
        om3_list.append(om3)
        t3_list.append(t3)

        v3_sampled[i, :] = sample_velocity(t3, v3, t_samp=t_sampled)
        print('Completed %i of %i variable time-step experiments' % (i+1, num_iterations))

    np.savez('v_ssa_bmax{:d}'.format(bond_max), *v3_list)
    np.savez('om_ssa_bmax{:d}'.format(bond_max), *om3_list)
    np.savez('t_ssa_bmax{:d}'.format(bond_max), *t3_list)

    v_avg = np.mean(v3_sampled, axis=0)
    v_std = np.std(v3_sampled, axis=0)

    if plot:
        plt.plot(t_sampled, v_avg, 'k', t_sampled[1:], (v_avg + v_std)[1:], 'r--',
                 t_sampled[1:], (v_avg - v_std)[1:], 'r--')
        plt.show()
    return None


stochastic_comparison(bond_max=10)
