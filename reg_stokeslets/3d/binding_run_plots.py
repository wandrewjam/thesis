import numpy as np
import matplotlib.pyplot as plt


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


def plot_trajectories(data_list):
    fig, ax = plt.subplots()
    for data in data_list:
        ax.plot(data['t'], data['z'], color='k', linewidth=.5)
    plt.show()


def main(runner):
    expt_names = extract_run_files(runner)
    data = extract_data(expt_names)
    plot_trajectories(data)
    print(expt_names)


if __name__ == '__main__':
    import sys

    main(sys.argv[1])
