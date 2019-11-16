import numpy as np
import matplotlib.pyplot as plt


def _get_steps_dwells(t, traj, thresh=20, samples=10000):
    # How is the threshold defined? Right now, it is just picked to get a ''reasonable'' number of steps and dwells
    # How is the number of samples defined? Right now, pretty arbitrary
    # Dwells are already defined so that the final dwell isn't counted
    skip = t.size // samples
    traj_sampled, t_sampled = traj[::skip], t[::skip]
    state = ((traj_sampled[1:] - traj_sampled[:-1])
             / (t_sampled[1:] - t_sampled[:-1])) < thresh
    state = np.append([False], state)
    trans = state[1:] != state[:-1]
    trans = np.append(trans, False)

    y_temp = traj_sampled[trans]
    t_temp = t_sampled[trans]
    steps = y_temp[2::2] - y_temp[:-2:2]
    dwell = t_temp[1::2] - t_temp[:-1:2]

    return steps, dwell


def main(filename):
    data = np.load(filename)
    vels = data['v_array']
    t = data['t_sample']

    trajs = np.cumsum(vels[:, :-1] * (t[1:] - t[:-1]), axis=1)
    trajs = np.insert(trajs, 0, 0, axis=1)

    # plt.plot(t, trajs.T)
    # plt.show()

    steps_dwells = [_get_steps_dwells(t, traj) for traj in trajs]
    steps, dwells = zip(*steps_dwells)

    steps = np.concatenate(steps)
    dwells = np.concatenate(dwells)
    avg_vels = trajs[:, -1] / t[-1]

    plt.hist(steps, density=True)
    plt.show()

    plt.hist(dwells, density=True)
    plt.show()

    plt.hist(avg_vels, density=True)
    plt.show()


if __name__ == '__main__':
    import sys

    filename = sys.argv[1]
    main(filename)