# This is a script to compare outputs from the different deterministic
# algorithms, applied to the fixed rolling problem with no translation

import numpy as np
import matplotlib.pyplot as plt

M = 512
up_data = np.load('./data/mov_rxns/multimov_pdeup_M{:d}_N{:d}_v0_om5_201118.npz'
                  .format(M, M))
bw_data = np.load('./data/mov_rxns/multimov_pdebw_M{:d}_N{:d}_v0_om5_101218.npz'
                  .format(M, M))
bn_data = np.load('./data/mov_rxns/multimov_bins_pde_M{:d}_N{:d}_v0_om5_251118'
                  '.npz'.format(M, M))

up_count = up_data['pde_count']
bw_count = bw_data['pde_count']
bn_count = bn_data['bin_count']
nu = np.pi/M

tp = up_data['tp']


def rel_err(count):
    return (count - bw_count)/bw_count[-1]


plt.plot(tp, rel_err(up_count), label='Eulerian upwind')
plt.plot(tp, rel_err(bn_count*nu), label='Semi-lagrangian', alpha=.5)
plt.plot(tp, np.zeros(shape=tp.shape), 'k', label='Reference line')
plt.xlabel('Nondimensional time')
plt.ylabel('Error relative to Eulerian Beam-Warming scheme')
plt.legend()
plt.show()

plt.plot(tp, up_count, label='Eulerian upwind')
plt.plot(tp, bn_count*nu, label='Semi-lagrangian')
plt.plot(tp, bw_count, 'k', label='Eulerian Beam-Warming')
plt.xlabel('Nondimensional time')
plt.ylabel('Bond quantity')
plt.legend()
plt.show()
