import numpy as np
import matplotlib.pyplot as plt
import glob


def process_trajectory(trajectory):
    """ Extract transition locations and times from trajectory """
    state = trajectory[1:, 1] == trajectory[:-1, 1]  # Size N - 1?
    state = np.append([False], state)  # Assume the initial state is a step
    trans = state[1:] != state[:-1]
    trans = np.append(trans, False)  # Assume no transition at the final time
    trans[0] = True  # Assume the initial time is a transition (to save point)
    trans[-1] = True  # Save the final point as well

    y_save = trajectory[trans, 1]  # Save the location of each state transition
    t_save = trajectory[trans, 0] - trajectory[0, 0]  # Save the time " " " "

    return state, y_save, t_save


def extract_state_data(t_save, y_save):
    pre_dwell_steps = None
    post_dwell_steps = None
    pre_free_vels = None
    post_free_vels = None

    del_y = y_save[1:] - y_save[:-1]
    del_t = t_save[1:] - t_save[:-1]

    if del_y[0] == 0 and del_y[-1] == 0:  # Dwell at begin and end of traj.
        steps = del_y[1::2]
        dwells = del_t[::2]
        free_vels = del_y[1::2] / del_t[1::2]
        vels = (y_save[-1] - y_save[0]) / (t_save[-1] - t_save[0])

    elif del_y[0] == 0:  # Dwell at begin of traj.
        steps = del_y[1:-1:2]
        dwells = del_t[::2]
        free_vels = del_y[1:-1:2] / del_t[1:-1:2]
        post_free_vels = del_y[-1] / del_t[-1]
        post_dwell_steps = del_y[-1]
        vels = (y_save[-2] - y_save[0]) / (t_save[-2] - t_save[0])

    elif del_y[-1] == 0:  # Dwell at end of traj.
        steps = del_y[2::2]
        dwells = del_t[1::2]
        pre_dwell_steps = del_y[0]
        free_vels = del_y[2::2] / del_t[2::2]
        pre_free_vels = del_y[0] / del_t[0]
        vels = (y_save[-1] - y_save[1]) / (t_save[-1] - t_save[1])

    else:  # Steps at begin and end of traj.
        steps = del_y[2:-1:2]
        dwells = del_t[1::2]
        pre_dwell_steps = del_y[0]
        post_dwell_steps = del_y[-1]
        free_vels = del_y[2:-1:2] / del_t[2:-1:2]
        pre_free_vels = del_y[0] / del_t[0]
        post_free_vels = del_y[-1] / del_t[-1]
        vels = (y_save[-2] - y_save[1]) / (t_save[-2] - t_save[1])

    assert dwells.size - steps.size == 1
    assert dwells.size - free_vels.size == 1
    if steps.size > 0:
        step_size_v_predwell = zip(steps, dwells[:-1])
        step_size_v_postdwell = zip(steps, dwells[1:])
        step_vel_v_predwell = zip(free_vels, dwells[:-1])
        step_vel_v_postdwell = zip(free_vels, dwells[1:])
        step_length_v_vel = zip(steps, free_vels)
        dwell_autocorr = zip(dwells[:-1], dwells[1:])
        step_autocorr = zip(steps[:-1], steps[1:])
        vels_autocorr = zip(free_vels[:-1], free_vels[1:])
    else:
        step_size_v_predwell = None
        step_size_v_postdwell = None
        step_vel_v_predwell = None
        step_vel_v_postdwell = None
        step_length_v_vel = None
        dwell_autocorr = None
        step_autocorr = None
        vels_autocorr = None
    return (dwells, free_vels, post_dwell_steps, post_free_vels,
            pre_dwell_steps, pre_free_vels, steps, vels, step_size_v_predwell,
            step_size_v_postdwell, step_vel_v_predwell, step_vel_v_postdwell,
            step_length_v_vel, dwell_autocorr, step_autocorr, vels_autocorr)


def main():
    experiments = dict()
    keys = ['hcp', 'ccp', 'hcw', 'ccw']
    for key in keys:
        trajectory_files = glob.glob(key + '-t*.dat')
        trajectory_list = [np.loadtxt(f) for f in trajectory_files]
        experiments.update([(key, trajectory_list)])

    expt = 'ccp'
    steps_combined = list()
    pre_dwell_steps_combined = list()
    post_dwell_steps_combined = list()
    dwells_combined = list()
    vels_combined = list()
    free_vels_combined = list()
    pre_free_vels_combined = list()
    post_free_vels_combined = list()
    for trajectory in experiments[expt]:
        state, y_save, t_save = process_trajectory(trajectory)

        (dwells, free_vels, post_dwell_steps, post_free_vels, pre_dwell_steps,
         pre_free_vels, steps, vels, step_size_v_predwell,
         step_size_v_postdwell, step_vel_v_predwell, step_vel_v_postdwell,
         step_length_v_vel, dwell_autocorr, step_autocorr, vels_autocorr) = (
            extract_state_data(t_save, y_save))

        steps_combined.append(steps)
        dwells_combined.append(steps)
        vels_combined.append(vels)
        free_vels_combined.append(free_vels)
        pre_dwell_steps_combined.append(pre_dwell_steps)
        post_dwell_steps_combined.append(post_dwell_steps)
        pre_free_vels_combined.append(pre_free_vels)
        post_free_vels_combined.append(post_free_vels)

    steps_combined = np.concatenate(steps_combined)
    dwells_combined = np.concatenate(dwells_combined)
    vels_combined = np.array(vels_combined)
    free_vels_combined = np.concatenate(free_vels_combined)
    pre_dwell_steps_combined = np.array([i for i in pre_dwell_steps_combined
                                         if i is not None])
    post_dwell_steps_combined = np.array([i for i in post_dwell_steps_combined
                                         if i is not None])
    pre_free_vels_combined = np.array([i for i in pre_free_vels_combined
                                         if i is not None])
    post_free_vels_combined = np.array([i for i in post_free_vels_combined
                                         if i is not None])

    print(np.sort(steps_combined))
    print(np.sort(pre_dwell_steps_combined))
    print(np.sort(post_dwell_steps_combined))
    print(np.sort(dwells_combined))
    plt.boxplot([steps_combined, pre_dwell_steps_combined, post_dwell_steps_combined],
                labels=['Steps between dwells', 'Steps entering domain',
                        'Steps leaving domain'])
    plt.ylabel('Step sizes $(\\mu m)$')
    plt.show()

    vels_combined = np.sort(vels_combined)
    plt.hist(vels_combined[vels_combined > 0])
    plt.show()

    np.savetxt(expt + '-vels.dat', vels_combined[vels_combined > 0])
    print(vels_combined)
    plt.hist(free_vels_combined, density=True)
    plt.hist(vels_combined, density=True)
    plt.xlabel('Velocity ($\\mu m / s$)')
    plt.legend(['Step velocities', 'Avg. velocities'])
    plt.show()

    print(free_vels_combined)

    plt.boxplot([free_vels_combined, pre_free_vels_combined, post_free_vels_combined],
                labels=['Vel. between dwells', 'Vel. entering domain',
                        'Vel. leaving domain'])
    plt.ylabel('Velocity ($\\mu m / s$)')
    plt.show()


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/vlado-data/'))

    main()
