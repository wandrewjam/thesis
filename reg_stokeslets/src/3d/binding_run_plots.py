import numpy as np
import matplotlib.pyplot as plt
import os
from motion_integration import get_bond_lengths
# import pdb


def get_steps_dwells(data_list):
    steps = []
    dwell = []
    dwell_maxes = []
    for data in data_list:
        state = []
        bond_counts = count_bonds(data)
        for i in range(len(data['t'])):
            state.append(np.any(data['bond_array'][:, 0, i] > -1))

        state = np.array(state)
        trans = state[1:] != state[:-1]
        trans = np.append([False], trans)
        i_trans = np.nonzero(trans)[0]

        split_counts = np.split(bond_counts, i_trans)
        bond_maxes = [np.amax(count) for count in split_counts]

        z_temp = data['z'][trans]
        t_temp = data['t'][trans]
        steps.append(z_temp[2::2] - z_temp[:-2:2])
        dwell.append(t_temp[1::2] - t_temp[:-1:2])
        dwell_maxes.append(bond_maxes[1:-1:2])

    return steps, dwell, dwell_maxes


def extract_run_files(runner):
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
    expt_names = []
    with open(txt_dir + runner + '.txt', 'r') as f:
        for line in f:
            expt_names.append(line[:-1])
    return expt_names


def extract_data(expt_names):
    data_dir = os.path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    data = []
    for expt in expt_names:
        try:
            with np.load(data_dir + expt + '.npz') as data_file:
                data_dict = {
                    't': data_file['t'], 'x': data_file['x'], 'y': data_file['y'],
                    'z': data_file['z'], 'r_matrices': data_file['r_matrices'],
                    'bond_array': data_file['bond_array'], 'num': expt[-3:],
                    'receptors': data_file['receptors']
                }
                if data_dict['t'][-1] > 3.0001:
                    data_dict['t'] *= .01
            data.append(data_dict)
        except FileNotFoundError:
            continue
    return data


def count_bonds(data):
    bonds = data['bond_array']
    counts = np.count_nonzero(bonds[:, 0, :] > -1, axis=0)
    return counts


def plot_trajectories(data_list, runner, save_plots=False):
    subsets = 8
    # assert len(data_list) % subsets == 0
    for i in range(len(data_list) // subsets):
        data_subset = data_list[subsets * i:subsets * (i + 1)]
        filename = runner + '_plot' + str(i)
        plot_trajectory_subset(data_subset, filename, save_plots=save_plots)


def plot_trajectory_subset(data_list, filename, save_plots=False):
    plot_dir = os.path.expanduser('~/thesis/reg_stokeslets/plots/')
    fig, ax = plt.subplots(nrows=3, sharex='all', figsize=(5, 9))
    ax_tw = ax[0].twinx()
    lw = 0.7
    for data in data_list:
        t, z = data['t'], data['z']
        if data['r_matrices'].shape[-1] == 3:
            e_mz = data['r_matrices'][:, 2, 0]
        else:
            e_mz = data['r_matrices'][2, 0, :]
        z_velocities = (z[1:] - z[:-1]) / (t[1:] - t[:-1])

        ax[0].plot(t, z, label=data['num'])
        ax_tw.plot(t[1:], z_velocities, linestyle='--', linewidth=lw)
        ax[1].plot(t, e_mz)

        # Count bonds and plot counts
        bond_counts = count_bonds(data)
        ax[2].plot(t, bond_counts)
    ax_tw.set_ybound(upper=185, lower=-50)
    ax[0].set_ylabel('Downstream displacement ($\\mu m$)')
    ax[0].legend()
    ax_tw.set_ylabel('$z$ velocities ($\\mu m / s$)')
    ax[1].set_ylabel('Orientation (z-cmp of $e_m$)')
    ax[2].set_ylabel('# of bonds')
    ax[2].set_xlabel('Time (s)')
    if save_plots:
        plt.savefig(plot_dir + filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def filter_traj(data_list):
    filtered = []
    for data in data_list:
        if data['z'][-1] > 150:
            filtered.append(data)
    return filtered


def get_average_vels(data_list):
    average_vels = []
    for data in data_list:
        if data['bond_array'].shape[0] > 0:
            # pdb.set_trace()
            average_vels.append(data['z'][-1] / data['t'][-1])
    return average_vels


def get_proportion_bound(data_list):
    num_bound = 0.
    for data in data_list:
        if data['bond_array'].shape[0] > 0:
            num_bound += 1
    return num_bound / len(data_list)


def get_bound_at_end(data_list):
    num_bound = 0.
    for data in data_list:
        if np.any(data['bond_array'][:, 0, -1] > -1):
            num_bound += 1
    return num_bound / len(data_list)


def extract_bond_information(data_list):
    lifetimes = []
    length_at_fm = []
    length_at_bk = []

    for data in data_list:
        formation_times = {}

        bond_array = data['bond_array']
        reaction_markers = np.nonzero(np.any(
            bond_array[:, 0, 1:] != bond_array[:, 0, :-1], axis=0))
        for time in reaction_markers[0]:
            old_bonds = bond_array[bond_array[:, 0, time] > -1, 0, time]
            new_bonds = bond_array[bond_array[:, 0, time+1] > -1, 0, time+1]
            # Ignoring a corner case where a new bond immediately forms
            # to the same receptor that was previously occupied

            broken_recs = np.setdiff1d(old_bonds, new_bonds,
                                       assume_unique=True)
            bonds_broken = bond_array[np.isin(bond_array[:, 0, time],
                                              broken_recs), :, time]

            for bd in bonds_broken:
                form_time = formation_times.pop(bd[0])
                lifetimes = np.append(lifetimes, data['t'][time+1] - form_time)

            formed_recs = np.setdiff1d(new_bonds, old_bonds,
                                       assume_unique=True)
            bonds_formed = bond_array[np.isin(bond_array[:, 0, time+1],
                                              formed_recs), :, time+1]

            for bd in bonds_formed:
                formation_times.update([(bd[0], data['t'][time])])

            # Check these bonds' lengths and add them to the lists
            if data['r_matrices'].shape[-1] == 3:
                data['r_matrices'] = data['r_matrices'].transpose(1, 2, 0)
            rmat = data['r_matrices'][:, :, time]
            x1, x2, x3 = data['x'][time], data['y'][time], data['z'][time]
            rmat_new = data['r_matrices'][:, :, time+1]
            x1n, x2n, x3n = data['x'][time+1], data['y'][time+1], data['z'][time+1]

            true_receptors = np.dot(data['receptors'], rmat.T)
            true_receptors += np.array([[x1, x2, x3]])

            true_receptors_new = np.dot(data['receptors'], rmat_new.T)
            true_receptors_new += np.array([[x1n, x2n, x3n]])

            broken_lengths = get_bond_lengths(bonds_broken,
                                              receptors=true_receptors_new)
            formed_lengths = get_bond_lengths(bonds_formed,
                                              receptors=true_receptors_new)

            length_at_fm = np.append(length_at_fm, formed_lengths)
            length_at_bk = np.append(length_at_bk, broken_lengths)
    return lifetimes, length_at_bk, length_at_fm


def main():
    # pdb.set_trace();
    save_plots = True
    expt_num = '8'

    plot_dir = os.path.expanduser('~/thesis/reg_stokeslets/plots/')

    if expt_num == '1':
        runners = ['bd_runner03', 'bd_runner02', 'bd_runner01', 'bd_runner04']
    elif expt_num == '2':
        runners = ['bd_runner05', 'bd_runner06', 'bd_runner07', 'bd_runner08']
    elif expt_num == '3':
        runners = ['bd_runner09', 'bd_runner10', 'bd_runner11', 'bd_runner12']
    elif expt_num == '4':
        runners = ['bd_runner13', 'bd_runner14', 'bd_runner15', 'bd_runner16']
    elif expt_num == '5':
        runners = ['bd_runner1101', 'bd_runner1102', 'bd_runner1103',
                   'bd_runner1104']
    elif expt_num == '6':
        runners = ['bd_runner1109', 'bd_runner1111']
    elif expt_num == '7':
        # runners = ['bd_runner2106', 'bd_runner2105', 'bd_runner2107', 
        #            'bd_runner2101', 'bd_runner2102']
        runners = ['bd_runner2106', 'bd_runner2105']
    elif expt_num == '8':
        runners = ['bd_runner3101', 'bd_runner3102', 'bd_runner1106', 'bd_runner3103']
    elif expt_num == '9':
        runners = ['bd_runner1106', 'bd_runner4101']
    else:
        raise ValueError('expt_num is invalid')

    flattened_steps_list = []
    flattened_dwell_list = []
    step_count_list = []
    dwell_count_list = []
    avg_step_list = []
    avg_dwell_list = []
    step_err_list = []
    dwell_err_list = []
    avg_v_list = []
    flattened_maxes_list = []
    lifetimes_list = []
    bk_length_list = []
    fm_length_list = []

    for runner in runners:
        expt_names = extract_run_files(runner)
        data = extract_data(expt_names)
        plot_trajectories(data, runner, save_plots=save_plots)
        # plot_velocities(data, runner, save_plots=save_plots)

        avg_v_list.append(get_average_vels(data))
        proportion_bound = get_proportion_bound(data)
        bound_at_end = get_bound_at_end(data)

        print('{} proportion bound: {}'.format(runner, proportion_bound))
        print('{} bound at end: {}'.format(runner, bound_at_end))

        steps, dwells, dwell_maxes = get_steps_dwells(data)
        flattened_steps_list.append(np.concatenate(steps))
        flattened_dwell_list.append(np.concatenate(dwells))
        flattened_maxes_list.append(np.concatenate(dwell_maxes))

        lifetimes, length_at_bk, length_at_fm = extract_bond_information(data)
        lifetimes_list.append(lifetimes)
        bk_length_list.append(length_at_bk)
        fm_length_list.append(length_at_fm)

        step_count_list.append([len(trial) for trial in steps])
        dwell_count_list.append([len(trial) for trial in dwells])

        avg_step_list.append(np.mean(flattened_steps_list[-1]))
        avg_dwell_list.append(np.mean(flattened_dwell_list[-1]))

        step_err_list.append(np.std(flattened_steps_list[-1]) / np.sqrt(flattened_steps_list[-1].shape[0]))
        dwell_err_list.append(np.std(flattened_dwell_list[-1]) / np.sqrt(flattened_dwell_list[-1].shape[0]))
        print('Finished with {}'.format(runner))

    if expt_num in ['1', '2', '3', '4', '5'] :
        labels = ['$k_{on} = 1$', '$k_{on} = 5$', '$k_{on} = 10$',
        '$k_{on} = 25$']
        # labels = ['$k_{on} = 1$', '$k_{on} = 5$', '$k_{on} = 10$']
    # labels = ['$k_{on} = 1$', '$k_{on} = 10$']
    elif expt_num == '6':
        labels = ['$k_{on} = 1$', '$k_{on} = 5$']
    elif expt_num == '7':
        # labels = ['$k_{on} = 0.05$', '$k_{on} = 0.1$',
        #           '$k_{on} = 0.5$', '$k_{on} = 1$', '$k_{on} = 5$']
        labels = ['$k_{on} = 0.05$', '$k_{on} = 0.1$']
    elif expt_num == '8':
        labels = ['$k_{off} = 1$', '$k_{off} = 2.5$',
        '$k_{off} = 5$', '$k_{off} = 10$']
        # labels = ['$k_{off} = 2.5$', '$k_{off} = 10$']
    elif expt_num == '9':
        labels = ['Normal receptors', 'Double receptors']
    plt.hist(avg_v_list, density=True)
    plt.xlabel('Average velocity ($\\mu m / s$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'avg_vels_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.hist(flattened_steps_list, density=True)
    plt.xlabel('Step size ($\\mu m$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'steps_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.hist(flattened_dwell_list, density=True)
    plt.xlabel('Pause time ($s$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'dwells_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    max_step_count = np.amax(np.concatenate(step_count_list))
    step_counts = [np.bincount(count, minlength=max_step_count+1) for count in step_count_list]
    width = 0.1
    x = np.arange(max_step_count+1)
    for i, counts in enumerate(step_counts):
        j = 2 * i - 3
        plt.bar(x + j*width/2, counts, width, label=labels[i])
    plt.xticks(x)
    plt.xlabel('Number of steps')
    plt.legend()
    if save_plots:
        plt.savefig(plot_dir + 'step_count_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    max_dwell_count = np.amax(np.concatenate(dwell_count_list))
    dwell_counts = [np.bincount(count, minlength=max_dwell_count + 1) for count in dwell_count_list]
    width = 0.1
    x = np.arange(max_dwell_count + 1)
    for i, counts in enumerate(dwell_counts):
        j = 2 * i - 3
        plt.bar(x + j * width / 2, counts, width, label=labels[i])
    plt.xticks(x)
    plt.xlabel('Number of dwells')
    plt.legend()
    if save_plots:
        plt.savefig(plot_dir + 'dwell_count_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.bar(np.arange(len(avg_v_list)), 
            [np.mean(el) for el in avg_v_list], 
            yerr=[
                np.std(el) / np.sqrt(128) for el in avg_v_list
            ], 
            tick_label=labels) 
    plt.ylabel('Mean of Average Velocity ($\\mu m / s$)')
    if save_plots:
        plt.savefig(plot_dir + 'avel_avg_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        
    plt.bar(np.arange(len(avg_step_list)), 
            avg_step_list, 
            yerr=step_err_list, 
            tick_label=labels)
    plt.ylabel('Average step size ($\\mu m$)')
    if save_plots:
        plt.savefig(plot_dir + 'step_avg_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.bar(np.arange(len(avg_dwell_list)), 
            avg_dwell_list, 
            yerr=dwell_err_list, 
            tick_label=labels)
    plt.ylabel('Average dwell time ($s$)')
    if save_plots:
        plt.savefig(plot_dir + 'dwell_avg_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    # print(expt_names)
    # print(proportion_bound)
    # print(bound_at_end)

    for dwells, maxes in zip(flattened_dwell_list, flattened_maxes_list):
        plt.plot(dwells, maxes, 'o')
    plt.xlabel('Dwell time (s)')
    plt.ylabel('Maximum bond count')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'dwell_corrs_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.hist(bk_length_list, density=True)
    plt.xlabel('Bond length at break ($\\mu m$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'bdbk_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.hist(fm_length_list, density=True)
    plt.xlabel('Bond length at formation ($\\mu m$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'bdfm_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.hist(lifetimes_list, density=True)
    plt.xlabel('Bond lifetimes ($s$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'bdlf_' + expt_num, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
