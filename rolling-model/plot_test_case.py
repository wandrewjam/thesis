import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    trials = int(raw_input('trials: '))
    M = int(raw_input('M: '))
    N = int(raw_input('N: '))
    v = float(raw_input('v: '))
    om = float(raw_input('om: '))

    filename = raw_input('filename: ')
    load_data = np.load('./data/mov_rxns/'+filename)
    bond_max = int(raw_input('bond max: '))

    par_array = load_data['par_array']
    fixed_arr = load_data['fixed_array']
    var_arr = load_data['var_array']
    pde_count = load_data['pde_count']
    tp = load_data['tp']

    fixed_avg = np.mean(fixed_arr, axis=0)
    var_avg = np.mean(var_arr, axis=0)
    fixed_std = np.std(fixed_arr, axis=0)
    var_std = np.std(var_arr, axis=0)

    if filename[0] == 's':
        plt.plot(tp[1:], (fixed_avg/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed time step')
        plt.plot(tp[1:], (var_avg/bond_max - pde_count)[1:]/pde_count[1:], 'g', label='Variable time step')
        plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'b:',
                 tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'b:', linewidth=.5)
        plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'g:',
                 tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))/bond_max - pde_count)[1:]/pde_count[1:], 'g:', linewidth=.5)
        plt.plot(tp, np.zeros(shape=tp.shape), 'k')
        plt.ylim((-.02, .02))
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

    elif filename[0] == 'm':
        nu = np.pi/N

        plt.plot(tp[1:], (fixed_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b', label='Fixed time step')
        plt.plot(tp[1:], ((fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b:',
                 tp[1:], ((fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'b:',
                 linewidth=.5)

        plt.plot(tp[1:], (var_avg*nu/bond_max - pde_count)[1:]/pde_count[1:], 'g', label='Variable time step')
        plt.plot(tp[1:], ((var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'g:',
                 tp[1:], ((var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max - pde_count)[1:]/pde_count[1:], 'g:',
                 linewidth=.5)

        if 'bin_count' in load_data.keys():
            bin_count = load_data['bin_count']
            plt.plot(tp[1:], (bin_count*nu - pde_count)[1:]/pde_count[1:], 'r', label='PDE with bins')
        plt.plot(tp, np.zeros(shape=tp.shape), 'k', label='Reference line')
        plt.ylim((-.02, .02))
        plt.legend(loc='best')
        plt.xlabel('Nondimensional time')
        plt.ylabel('Relative error')
        plt.show()

        plt.plot(tp, fixed_avg*nu/bond_max, 'b', label='Fixed time step')
        plt.plot(tp, (fixed_avg + 2*fixed_std/np.sqrt(trials))*nu/bond_max, 'b:',
                 tp, (fixed_avg - 2*fixed_std/np.sqrt(trials))*nu/bond_max, 'b:', linewidth=.5)

        plt.plot(tp, var_avg*nu/bond_max, 'g', label='Variable time step')
        plt.plot(tp, (var_avg + 2*var_std/np.sqrt(trials))*nu/bond_max, 'g:',
                 tp, (var_avg - 2*var_std/np.sqrt(trials))*nu/bond_max, 'g:', linewidth=.5)

        if 'bin_count' in load_data.keys():
            plt.plot(tp, bin_count*nu, 'r', label='PDE with bins')
        plt.plot(tp, pde_count, 'k', label='Deterministic')
        plt.legend(loc='best')
        plt.xlabel('Nondimensional time')
        plt.ylabel('Bond quantity')
        plt.show()
