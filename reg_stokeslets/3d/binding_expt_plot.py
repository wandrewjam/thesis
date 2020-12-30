import matplotlib.pyplot as plt
import numpy as np
from motion_integration import (get_bond_lengths, find_min_separation,
                                repulsive_force)


def main(filename):
    load_dir = 'data/'
    plot_dir = 'plots/'
    # file_fmt = 'bd_expt{:03d}'.format(expt_num)

    load_data = np.load(load_dir + filename + '.npz')
    t, x, y, z = load_data['t'], load_data['x'], load_data['y'], load_data['z']
    r_matrices = load_data['r_matrices']
    receptors = load_data['receptors']
    bond_array = load_data['bond_array']
    mask = x > 0
    fig, axs = plt.subplots(5, sharex='all', figsize=(6.5, 9))
    x_ln = axs[0].plot(t[mask], x[mask], label='x')
    ax_tw = axs[0].twinx()
    z_ln = ax_tw.plot(t[mask], z[mask], color='tab:orange', label='z')
    axs[0].set_ylabel('Center of mass ($\\mu$m)')
    lns = x_ln + z_ln
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs)
    axs[1].plot(t[mask], r_matrices[2, 0, mask])
    axs[1].set_ylabel('Orientation')
    axs[1].legend(['e_mz'])
    center = np.stack([x, y, z])
    # fig, ax = plt.subplots()
    # if expt == 3:
    # Figure out how to plot bonds individually (and only bonds that exist at a particular time point)
    # Use a dict to track individual bonds over time
    bond_dict = {}
    for i, t_i in enumerate(t):
        rmat = r_matrices[:, :, i]
        true_receptors = center[:, [i]].T + np.dot(receptors, rmat.T)
        current_mask = bond_array[:, 0, i] != -1
        current_bonds = bond_array[current_mask, :, i]
        bond_lens = get_bond_lengths(current_bonds, true_receptors)
        z_rec = true_receptors[current_bonds[:, 0].astype('int'), 2] - z[i]
        z_cmp = current_bonds[:, 2] - true_receptors[
            current_bonds[:, 0].astype('int'), 2]
        x_cmp = true_receptors[current_bonds[:, 0].astype('int'), 0]
        y_cmp = current_bonds[:, 1] - true_receptors[
            current_bonds[:, 0].astype('int'), 1]

        for j in range(len(current_bonds)):
            key = tuple(current_bonds[j, :])
            try:
                bond_dict[key] = (
                    np.append(bond_dict[key][0], t_i),
                    np.append(bond_dict[key][1], bond_lens[j]),
                    np.append(bond_dict[key][2], z_rec[j]),
                    np.append(bond_dict[key][3], z_cmp[j]),
                    np.append(bond_dict[key][4], x_cmp[j]),
                    np.append(bond_dict[key][5], y_cmp[j]))
            except KeyError:
                bond_dict[key] = (t_i, bond_lens[j], z_rec[j], z_cmp[j],
                                  x_cmp[j], y_cmp[j])
    for times, bond_len, z_rec, z_cmp, x_cmp, y_cmp in bond_dict.values():
        line, = axs[2].plot(times, bond_len)
        axs[3].plot(times, z_rec, color=line.get_color())
        axs[4].plot(times, z_cmp, color=line.get_color())
    # else:
    #     lengths = np.linalg.norm(
    #         center[:, None, :] + np.dot(result[3].transpose((2, 0, 1)),
    #                                     receptors.T).transpose((1, 2, 0)),
    #         axis=0)
    #     axs[2].plot(t[mask], lengths.T[mask])
    # axs[2].set_xlim(t[0], t[-1])
    # axs[2].set_title('Bond length')
    # axs[2].set_xlabel('Time (s)')
    # Set horizontal reference line
    t_ref = [t[0], t[-1]]
    axs[2].plot(t_ref, [0.1, 0.1], linewidth=.7, color='k')
    axs[3].plot(t_ref, [0, 0], linewidth=.7, color='k')
    axs[4].plot(t_ref, [0, 0], linewidth=.7, color='k')

    axs[2].set_ylabel('Length ($\\mu$m)')
    axs[3].set_ylabel('$z$ difference ($\\mu$m)')
    axs[4].set_ylabel('$z$ difference ($\\mu$m)')
    axs[-1].set_xlabel('Time (s)')
    # plt.savefig(plot_dir + filename + '_1', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # Compute separation distance and rep force for each time step
    sep_distance = np.zeros(shape=x.shape)
    rep_force = np.zeros(shape=x.shape)
    for i in range(len(x)):
        ix, iy, iz = x[i], y[i], z[i]
        irmat = r_matrices[:, :, i]
        sep_distance[i] = find_min_separation(ix, irmat[:, 0])
        rep_force[i] = repulsive_force(ix, irmat[:, 0])[1]
    fig, axs = plt.subplots(nrows=3)
    axs[0].semilogy(t, sep_distance)
    axs[1].plot(t, rep_force)
    for times, bond_len, z_rec, z_cmp, x_cmp, y_cmp in bond_dict.values():
        axs[2].plot(times, x_cmp)
        axs[2].plot(times, y_cmp, linestyle='--')
    # plt.savefig(plot_dir + filename + '_2', bbox_inches='tight')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # for num in range(9):
    #     main(num)
    import sys

    main(sys.argv[1])
