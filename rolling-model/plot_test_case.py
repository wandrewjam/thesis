import numpy as np
import matplotlib.pyplot as plt

trials = int(raw_input('trials: '))
filename = raw_input('filename: ')
load_data = np.load('./data/sta_rxns/'+filename)

par_array = load_data['par_array']
fixed_arr = load_data['fixed_array']
var_arr = load_data['var_array']
pde_count = load_data['pde_count']
tp = load_data['tp']

bond_max = par_array[6]

fixed_avg = np.mean(fixed_arr, axis=0)
var_avg = np.mean(var_arr, axis=0)
fixed_std = np.std(fixed_arr, axis=0)
var_std = np.std(var_arr, axis=0)

plt.plot(tp[1:], (fixed_avg/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed time step')
plt.plot(tp[1:], (var_avg/bond_max - pde_count)[1:]/pde_count[1:], 'g', label='Variable time step')
plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'b:',
         tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'b:', linewidth=.5)
plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'g:',
         tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'g:', linewidth=.5)
plt.plot(tp, np.zeros(shape=tp.shape), 'k')
plt.legend(loc='best')
plt.title('Relative errors for the stochastic algorithms for a single receptor')
plt.show()

plt.plot(tp, fixed_avg/bond_max, 'b', label='Fixed time step')
plt.plot(tp, var_avg/bond_max, 'g', label='Variable time step')
plt.plot(tp, (fixed_avg + 2*fixed_std/np.sqrt(trials))/bond_max, 'b:',
         tp, (fixed_avg - 2*fixed_std/np.sqrt(trials))/bond_max, 'b:', linewidth=.5)
plt.plot(tp, (var_avg + 2*var_std/np.sqrt(trials))/bond_max, 'g:',
         tp, (var_avg - 2*var_std/np.sqrt(trials))/bond_max, 'g:', linewidth=.5)
plt.plot(tp, pde_count, 'k', label='Deterministic')
plt.legend(loc='best')
plt.title('Bond numbers for each algorithm for a single receptor')
plt.show()
