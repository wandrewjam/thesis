import numpy as np
import matplotlib.pyplot as plt
import os


def get_steps_dwells(data_list):
    steps = []
    dwell = []
    for data in data_list:
        state = []
        for i in range(len(data['t'])):
            state.append(np.any(data['bond_array'][:, 0, i] > -1))
        state = np.array(state)
        trans = state[1:] != state[:-1]
        trans = np.append([False], trans)

        z_temp = data['z'][trans]
        t_temp = data['t'][trans]
        steps.append(z_temp[2::2] - z_temp[:-2:2])
        dwell.append(t_temp[1::2] - t_temp[:-1:2])

    return steps, dwell


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
        with np.load(data_dir + expt + '.npz') as data_file:
            data_dict = {
                't': data_file['t'], 'x': data_file['x'], 'y': data_file['y'],
                'z': data_file['z'], 'r_matrices': data_file['r_matrices'],
                'bond_array': data_file['bond_array'], 'num': expt[-3:]
            }
        data.append(data_dict)
    return data


def count_bonds(data):
    bonds = data['bond_array']
    counts = np.count_nonzero(bonds[:, 0, :] > -1, axis=0)
    return counts


def plot_trajectories(data_list, runner, save_plots=False):
    subsets = 8
    assert len(data_list) % subsets == 0
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
        e_mz = data['r_matrices'][2, 0, :]
        z_velocities = (z[1:] - z[:-1]) / (t[1:] - t[:-1])

        ax[0].plot(t, z, label=data['num'])
        ax_tw.plot(t[1:], z_velocities, linestyle='--', linewidth=lw)
        ax[1].plot(t, e_mz)

        # Count bonds and plot counts
        bond_counts = count_bonds(data)
        ax[2].plot(t, bond_counts)
    ax_tw.set_ybound(upper=185)
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


def main():
    save_plots = True
    plot_dir = os.path.expanduser('~/thesis/reg_stokeslets/plots/')
    # runners = ['bd_runner03', 'bd_runner02', 'bd_runner01', 'bd_runner04']
    runners = ['bd_runner03', 'bd_runner02', 'bd_runner01']
    steps_list = []
    dwell_list = []
    avg_v_list = []
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

        steps, dwells = get_steps_dwells(data)
        steps_list.append(np.concatenate(steps))
        dwell_list.append(np.concatenate(dwells))
        print('Finished with {}'.format(runner))

    # labels = ['$k_{on} = 1$', '$k_{on} = 5$', '$k_{on} = 10$', '$k_{on} = 25$']
    labels = ['$k_{on} = 1$', '$k_{on} = 5$', '$k_{on} = 10$']
    plt.hist(avg_v_list, density=True)
    plt.xlabel('Average velocity ($\\mu m / s$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'avg_vels', bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.hist(steps_list, density=True)
    plt.xlabel('Step size ($\\mu m$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'steps', bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    plt.hist(dwell_list, density=True)
    plt.xlabel('Pause time ($s$)')
    plt.legend(labels)
    if save_plots:
        plt.savefig(plot_dir + 'dwells', bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    # print(expt_names)
    # print(proportion_bound)
    # print(bound_at_end)


if __name__ == '__main__':
    main()
