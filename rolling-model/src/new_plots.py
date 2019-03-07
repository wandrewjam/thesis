import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    import sys
    det_loadname, sto_loadname = sys.argv[1], sys.argv[2]

    det_data = np.load(det_loadname)
    sto_data = np.load(sto_loadname)

    bond_counts, v, om, avg_lengths, bond_mesh, force, torque, t_mesh = (
        det_data['bond_counts'], det_data['v'], det_data['om'],
        det_data['avg_lengths'], det_data['bond_mesh'], det_data['force'],
        det_data['torque'], det_data['t_mesh']
    )

    (count_mean, count_std, v_mean, v_std, om_mean, om_std, force_mean,
     force_std, torque_mean, torque_std, len_mean, len_std, master_list,
     t_sample, count_sampled, v_sampled, om_sampled, fc_sampled, tq_sampled,
     len_sampled) = (
        sto_data['count_mean'], sto_data['count_std'], sto_data['v_mean'],
        sto_data['v_std'], sto_data['om_mean'], sto_data['om_std'],
        sto_data['force_mean'], sto_data['force_std'], sto_data['torque_mean'],
        sto_data['torque_std'], sto_data['len_mean'], sto_data['len_std'],
        sto_data['master_list'], sto_data['t_sample'],
        sto_data['count_sampled'], sto_data['v_sampled'],
        sto_data['om_sampled'], sto_data['fc_sampled'], sto_data['tq_sampled'],
        sto_data['len_sampled']
    )

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='none',
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

    ax[0, 1].plot(t_mesh, force, 'r', t_sample, force_mean, 'b')
    ax[0, 1].plot(t_sample, force_mean + force_std, 'b--', t_sample,
                  force_mean - force_std, 'b--', linewidth=.5)
    ax[0, 1].set_ylabel('Net horizontal force')
    ax[1, 1].plot(t_mesh, torque, 'r', t_sample, torque_mean, 'b')
    ax[1, 1].plot(t_sample, torque_mean + torque_std, 'b--', t_sample,
                  torque_mean - torque_std, 'b--', linewidth=.5)
    ax[1, 1].set_ylabel('Net torque')
    ax[2, 1].plot(t_mesh, avg_lengths, 'r', t_sample[1:], len_mean, 'b')
    ax[2, 1].plot(t_sample[1:], len_mean + len_std, 'b--', t_sample[1:],
                  len_mean - len_std, 'b--', linewidth=.5)
    ax[2, 1].set_xlabel('Nondimensional time')
    ax[2, 1].set_ylabel('Average bond length')
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='row',
                             figsize=(12, 8))

    lw = 1
    runs = count_sampled.shape[0]

    ax1[0, 0].plot(t_sample, np.transpose(v_sampled[:runs/3]), linewidth=lw)
    ax1[0, 0].set_ylabel('Translation velocity ($v$)')
    ax1[1, 0].plot(t_sample, np.transpose(om_sampled[:runs/3]), linewidth=lw)
    ax1[1, 0].set_ylabel('Rotation velocity ($\\omega$)')
    ax1[2, 0].plot(t_sample, np.transpose(count_sampled[:runs/3]), linewidth=lw)
    ax1[2, 0].set_xlabel('Nondimensional time')
    ax1[2, 0].set_ylabel('Bond number')

    ax1[0, 1].plot(t_sample, np.transpose(v_sampled[runs/3:2*runs/3]), linewidth=lw)
    ax1[1, 1].plot(t_sample, np.transpose(om_sampled[runs/3:2*runs/3]), linewidth=lw)
    ax1[2, 1].plot(t_sample, np.transpose(count_sampled[runs/3:2*runs/3]), linewidth=lw)
    ax1[0, 2].plot(t_sample, np.transpose(v_sampled[2*runs/3:]), linewidth=lw)
    ax1[1, 2].plot(t_sample, np.transpose(om_sampled[2*runs/3:]), linewidth=lw)
    ax1[2, 2].plot(t_sample, np.transpose(count_sampled[2*runs/3:]), linewidth=lw)
    ax1[2, 1].set_xlabel('Nondimensional time')
    ax1[2, 2].set_xlabel('Nondimensional time')
    plt.tight_layout()
    plt.show()
