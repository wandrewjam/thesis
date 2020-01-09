import numpy as np
import matplotlib.pyplot as plt
import glob


def main():
    experiments = dict()
    keys = ['hcp', 'ccp', 'hcw', 'ccw']
    for key in keys:
        trajectory_files = glob.glob(key + '-t*.dat')
        trajectory_list = [np.loadtxt(f) for f in trajectory_files]
        experiments.update([(key, trajectory_list)])

    expt = 'ccw'
    steps = list()
    pre_dwell_steps = list()
    post_dwell_steps = list()
    dwells = list()
    vels = list()
    free_vels = list()
    pre_free_vels = list()
    post_free_vels = list()
    for trajectory in experiments[expt]:
        state = trajectory[1:, 1] == trajectory[:-1, 1]
        state = np.append([False], state)
        trans = state[1:] != state[:-1]
        trans = np.append(trans, False)
        trans[0] = True
        trans[-1] = True

        y_temp = trajectory[trans, 1]
        t_temp = trajectory[trans, 0] - trajectory[0, 0]
        del_y = y_temp[1:] - y_temp[:-1]
        del_t = t_temp[1:] - t_temp[:-1]
        if del_y[0] == 0 and del_y[-1] == 0:
            steps = np.append(steps, del_y[1::2])
            dwells = np.append(steps, del_t[::2])
            free_vels = np.append(free_vels, del_y[1::2] / del_t[1::2])
            vels.append((y_temp[-1] - y_temp[0])/(t_temp[-1] - t_temp[0]))
        elif del_y[0] == 0:
            steps = np.append(steps, del_y[1:-1:2])
            dwells = np.append(dwells, del_t[::2])
            free_vels = np.append(free_vels, del_y[1:-1:2] / del_t[1:-1:2])
            post_free_vels = np.append(post_free_vels, del_y[-1] / del_t[-1])
            post_dwell_steps = np.append(post_dwell_steps, del_y[-1])
            vels.append((y_temp[-2] - y_temp[0])/(t_temp[-2] - t_temp[0]))
        elif del_y[-1] == 0:
            steps = np.append(steps, del_y[2::2])
            dwells = np.append(dwells, del_t[1::2])
            pre_dwell_steps = np.append(pre_dwell_steps, del_y[0])
            free_vels = np.append(free_vels, del_y[2::2] / del_t[2::2])
            pre_free_vels = np.append(pre_free_vels, del_y[0] / del_t[0])
            vels.append((y_temp[-1] - y_temp[1])/(t_temp[-1] - t_temp[1]))
        else:
            steps = np.append(steps, del_y[2:-1:2])
            dwells = np.append(dwells, del_t[1::2])
            pre_dwell_steps = np.append(pre_dwell_steps, del_y[0])
            post_dwell_steps = np.append(post_dwell_steps, del_y[-1])
            free_vels = np.append(free_vels, del_y[2:-1:2] / del_t[2:-1:2])
            pre_free_vels = np.append(pre_free_vels, del_y[0] / del_t[0])
            post_free_vels = np.append(post_free_vels, del_y[-1] / del_t[-1])
            vels.append((y_temp[-2] - y_temp[1]) / (t_temp[-2] - t_temp[1]))

    print(np.sort(steps))
    print(np.sort(pre_dwell_steps))
    print(np.sort(post_dwell_steps))
    print(np.sort(dwells))
    plt.boxplot([steps, pre_dwell_steps, post_dwell_steps],
                labels=['Steps between dwells', 'Steps entering domain',
                        'Steps leaving domain'])
    plt.ylabel('Step sizes $(\\mu m)$')
    plt.show()

    vels = np.sort(vels)
    plt.hist(vels[vels > 0])
    plt.show()

    np.savetxt(expt + '-vels.dat', vels[vels > 0])
    print(vels)
    plt.hist(free_vels, density=True)
    plt.hist(vels, density=True)
    plt.show()

    print(free_vels)

    plt.boxplot([free_vels, pre_free_vels, post_free_vels],
                labels=['Vel. between dwells', 'Vel. entering domain',
                        'Vel. leaving domain'])
    plt.ylabel('Velocity $(\\mu m / s)$')
    plt.show()


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/vlado-data/'))

    main()
