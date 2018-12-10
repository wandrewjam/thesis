import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_folder = './data/mov_rxns/'

    # Load PDE data
    data_pde = np.load(data_folder + 'multimov_pde_M512_N512_v0_om5_201118.npz')

    # Load fixed time step data
    data_fixed = np.load(data_folder + 'multimov_fixed_N512_v0_om5_trials10000_191118.npz')

    # Load variable time step data
    data_var = np.load(data_folder + 'multimov_var_N512_v0_om5_trials10000_261118.npz')

    # Load PDE bins data
    data_bins = np.load(data_folder + 'multimov_bins_pde_M512_N512_v0_om5_251118.npz')

    bond_max = 10
    trials = 10000
    N = 512

    tp = data_pde['tp']
    pde_count = data_pde['pde_count']
    bin_count = data_bins['bin_count']
    fixed_arr = data_fixed['fixed_array']
    var_arr = data_var['var_array']

    fixed_avg = np.mean(fixed_arr, axis=0)
    var_avg = np.mean(var_arr, axis=0)
    fixed_std = np.std(fixed_arr, axis=0)
    var_std = np.std(var_arr, axis=0)

    nu = np.pi/N

    plt.plot(tp[1:], (fixed_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed time step')
    plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b:',
             tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b:',
             linewidth=.5)

    plt.plot(tp[1:], (var_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'g', label='Variable time step')
    plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'g:',
             tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'g:',
             linewidth=.5)

    plt.plot(tp[1:], (bin_count*nu - pde_count)[1:]/pde_count[1:], 'r', label='PDE with bins')
    plt.plot(tp, np.zeros(shape=tp.shape), 'k', label='Reference line')

    plt.ylim((-.02, .02))
    plt.legend(loc='best')
    plt.xlabel('Nondimensional time')
    plt.ylabel('Relative error in bond number')
    plt.show()

    plt.plot(tp, fixed_avg*nu/bond_max, 'b', label='Fixed time step')
    plt.plot(tp, (fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max, 'b:',
             tp, (fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max, 'b:', linewidth=.5)

    plt.plot(tp, var_avg*nu/bond_max, 'g', label='Variable time step')
    plt.plot(tp, (var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max, 'g:',
             tp, (var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max, 'g:', linewidth=.5)

    plt.plot(tp, bin_count*nu, 'r', label='PDE with bins')
    plt.plot(tp, pde_count, 'k', label='Deterministic')
    plt.legend(loc='best')
    plt.xlabel('Nondimensional time')
    plt.ylabel('Bond quantity')
    plt.show()

