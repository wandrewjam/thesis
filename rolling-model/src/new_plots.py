import matplotlib.pyplot as plt
import numpy as np
from rolling_algorithms import load_deterministic_data, load_stochastic_data


if __name__ == '__main__':
    M, N = 64, 64
    time_steps = 5180*3
    init = 'free'
    bond_max = 100
    trials = 4
    correct_flux = False

    bond_counts, v, om, t_mesh = load_deterministic_data(M, N, time_steps,
                                                         init, scheme='bw')
    count_array, v_array, om_array, t_sample = load_stochastic_data(
        trials, M, N, time_steps, init, bond_max, correct_flux)

    bond_counts *= bond_max*N/np.pi
    count_array *= bond_max*N/np.pi

    count_mean, v_mean, om_mean = tuple(np.mean(a, axis=0) for a in
                                        (count_array, v_array, om_array))
    count_std, v_std, om_std = tuple(np.std(a, axis=0) for a in
                                     (count_array, v_array, om_array))

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='row',
                           figsize=(12, 8))

    ax[0, 0].plot(t_mesh, v, 'r', t_sample, v_mean, 'b')
    ax[0, 0].plot(t_sample, v_mean + v_std, 'b--', t_sample, v_mean - v_std,
               'b--', linewidth=.5)
    ax[0, 0].set_ylabel('Translation velocity ($v$)')
    ax[1, 0].plot(t_mesh, om, 'r', t_sample, om_mean, 'b')
    ax[1, 0].plot(t_sample, om_mean + om_std, 'b--', t_sample, om_mean - om_std,
               'b--', linewidth=.5)
    ax[1, 0].set_ylabel('Rotation rate ($\\omega$)')
    ax[2, 0].plot(t_mesh, bond_counts, 'r', t_sample, count_mean, 'b')
    ax[2, 0].plot(t_sample, count_mean + count_std, 'b--', t_sample,
               count_mean - count_std, 'b--', linewidth=.5)
    ax[2, 0].set_ylabel('Bond number')
    ax[2, 0].set_xlabel('Nondimensional time')

    ax[0, 1].plot(t_sample, np.transpose(v_array)[:, 1], 'b', linewidth=.5)
    ax[0, 1].plot(t_mesh, v, 'r')
    ax[1, 1].plot(t_sample, np.transpose(om_array)[:, 1], 'b', linewidth=.5)
    ax[1, 1].plot(t_mesh, om, 'r')
    ax[2, 1].plot(t_sample, np.transpose(count_array)[:, 1], 'b', linewidth=.5)
    ax[2, 1].plot(t_mesh, bond_counts, 'r')
    ax[2, 1].set_xlabel('Nondimensional time')
    plt.tight_layout()
    plt.show()
