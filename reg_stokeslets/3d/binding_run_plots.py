import numpy as np
import matplotlib.pyplot as plt


def get_steps_dwells(data_list, thresh=1, samples=15):
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
    import os
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/3d/par-files/')
    expt_names = []
    with open(txt_dir + runner + '.txt', 'r') as f:
        for line in f:
            expt_names.append(line[:-1])
    return expt_names


def extract_data(expt_names):
    import os
    data_dir = os.path.expanduser('~/thesis/reg_stokeslets/3d/data/')
    data = []
    for expt in expt_names:
        data.append(np.load(data_dir + expt + '.npz'))
    return data


def count_bonds(data):
    bonds = data['bond_array']
    counts = np.count_nonzero(bonds[:, 0, :] > -1, axis=0)
    return counts


def plot_trajectories(data_list):
    fig, ax = plt.subplots(nrows=2, sharex='all', figsize=(6, 8))
    for data in data_list:
        lw = .5
        c = 'C0'
        ax[0].plot(data['t'], data['z'], color=c, linewidth=lw)
        # ax[1].plot(data['t'], data['r_matrices'][2, 0, :], color=c,
        #            linewidth=lw)

        # Count bonds and plot counts
        bond_counts = count_bonds(data)
        ax[1].plot(data['t'], bond_counts, color=c, linewidth=lw)

    ax[0].set_ylabel('Downstream displacement ($\\mu m$)')
    # ax[1].set_ylabel('Orientation (z-cmp of $e_m$)')
    ax[1].set_ylabel('# of bonds')
    ax[1].set_xlabel('Time (s)')
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
    runners = ['bd_runner03', 'bd_runner02', 'bd_runner01', 'bd_runner04']
    # runners = ['bd_runner03']
    steps_list = []
    dwell_list = []
    avg_v_list = []
    for runner in runners:
        expt_names = extract_run_files(runner)
        data = extract_data(expt_names)
        plot_trajectories(data)

        avg_v_list.append(get_average_vels(data))
        proportion_bound = get_proportion_bound(data)
        bound_at_end = get_bound_at_end(data)

        print('{} proportion bound: {}'.format(runner, proportion_bound))
        print('{} bound at end: {}'.format(runner, bound_at_end))

        steps, dwells = get_steps_dwells(data)
        steps_list.append(np.concatenate(steps))
        dwell_list.append(np.concatenate(dwells))
        print('Finished with {}'.format(runner))

    labels = ['$k_{on} = 1$', '$k_{on} = 5$', '$k_{on} = 10$', '$k_{on} = 25$']
    plt.hist(avg_v_list)
    plt.xlabel('Average velocity ($\\mu m / s$)')
    plt.legend(labels)
    plt.show()

    plt.hist(steps_list)
    plt.xlabel('Step size ($\\mu m$)')
    plt.legend(labels)
    plt.show()

    plt.hist(dwell_list)
    plt.xlabel('Pause time ($s$)')
    plt.legend(labels)
    plt.show()
    # print(expt_names)
    # print(proportion_bound)
    # print(bound_at_end)


if __name__ == '__main__':
    main()
