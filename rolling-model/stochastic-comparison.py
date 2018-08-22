from stochastic_model_2 import *
from stochastic_model_ssa import *


def sample_velocity(t, v, t_samp=None):
    if t_samp is None:
        t_samp = np.linspace(start=0, stop=np.max(t), num=100)
        return v[np.searchsorted(t, t_samp)], t_samp

    return v[np.searchsorted(t, t_samp)]


num_iterations, num_samples = 100, 200
v_sampled = np.zeros(shape=(num_iterations, num_samples))
t_sampled = np.linspace(start=0, stop=0.4, num=num_samples)

v1, om1, t1 = stochastic_model_tau()

for i in range(num_iterations):
    v2, om2, t2, n = stochastic_model_ssa(bond_max=10)
    v_sampled[i, :] = sample_velocity(t2, v2, t_samp=t_sampled)
    print('Completed %i of %i experiments' % (i+1, num_iterations))

v_avg = np.mean(v_sampled, axis=0)
v_std = np.std(v_sampled, axis=0)

plt.plot(t_sampled, v_avg, 'k', t_sampled, v_avg + v_std, 'r--', t_sampled, v_avg - v_std, 'r--')
plt.show()



