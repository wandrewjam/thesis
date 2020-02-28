import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, expon
import glob


def process_trajectory(trajectory, absolute_pause_threshold=0.):
    """Extract transition locations and times from trajectory

    Parameters
    ----------
    trajectory
    absolute_pause_threshold

    Returns
    -------

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Look for errors in trajectory data where there are multiple
        # data points for a single time
        velocity = ((trajectory[1:, 1] - trajectory[:-1, 1])
                    / (trajectory[1:, 0] - trajectory[:-1, 0]))
    indices = np.nonzero(~np.isfinite(velocity))[0]
    if len(indices) > 0:
        end = indices[0]
        velocity = velocity[:end]
        trajectory = trajectory[:end+1]
    is_bound = velocity <= absolute_pause_threshold  # Size N - 1?
    is_bound = np.append([False], is_bound)  # Assume the initial state is a step
    # is_bound[i] stores the state in the interval (t[i-1]----t[i])

    trans = is_bound[1:] != is_bound[:-1]
    trans = np.append(trans, False)  # Assume no transition at the final time
    trans[0] = True  # Assume the initial time is a transition (to save point)
    trans[-1] = True  # Save the final point as well

    y_save = trajectory[trans, 1]  # Save the location of each state transition
    t_save = trajectory[trans, 0] - trajectory[0, 0]  # Save the time " " " "

    return is_bound, y_save, t_save


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
        if free_vels.size > 0:
            avg_free_vels = np.array(np.average(
                free_vels, weights=del_t[1::2]), ndmin=1)
        else:
            avg_free_vels = np.copy(free_vels)
        vels = (y_save[-1] - y_save[0]) / (t_save[-1] - t_save[0])

    elif del_y[0] == 0:  # Dwell at begin of traj.
        steps = del_y[1:-1:2]
        dwells = del_t[::2]
        free_vels = del_y[1:-1:2] / del_t[1:-1:2]
        if free_vels.size > 0:
            avg_free_vels = np.array(np.average(
                free_vels, weights=del_t[1:-1:2]), ndmin=1)
        else:
            avg_free_vels = np.copy(free_vels)
        post_free_vels = del_y[-1] / del_t[-1]
        post_dwell_steps = del_y[-1]
        vels = (y_save[-2] - y_save[0]) / (t_save[-2] - t_save[0])

    elif del_y[-1] == 0:  # Dwell at end of traj.
        steps = del_y[2::2]
        dwells = del_t[1::2]
        pre_dwell_steps = del_y[0]
        free_vels = del_y[2::2] / del_t[2::2]
        if free_vels.size > 0:
            avg_free_vels = np.array(np.average(
                free_vels, weights=del_t[2::2]), ndmin=1)
        else:
            avg_free_vels = np.copy(free_vels)
        pre_free_vels = del_y[0] / del_t[0]
        vels = (y_save[-1] - y_save[1]) / (t_save[-1] - t_save[1])

    else:  # Steps at begin and end of traj.
        steps = del_y[2:-1:2]
        dwells = del_t[1::2]
        pre_dwell_steps = del_y[0]
        post_dwell_steps = del_y[-1]
        free_vels = del_y[2:-1:2] / del_t[2:-1:2]
        if free_vels.size > 0:
            avg_free_vels = np.array(np.average(
                free_vels, weights=del_t[2:-1:2]), ndmin=1)
        else:
            avg_free_vels = np.copy(free_vels)
        pre_free_vels = del_y[0] / del_t[0]
        post_free_vels = del_y[-1] / del_t[-1]
        vels = (y_save[-2] - y_save[1]) / (t_save[-2] - t_save[1])

    assert dwells.size - steps.size == 1 or (dwells.size == 0
                                             and steps.size == 0)
    assert dwells.size - free_vels.size == 1 or (dwells.size == 0
                                                 and steps.size == 0)
    if steps.size > 0:
        step_size_v_predwell = zip(steps, dwells[:-1])
        step_size_v_postdwell = zip(steps, dwells[1:])
        step_vel_v_predwell = zip(free_vels, dwells[:-1])
        step_vel_v_postdwell = zip(free_vels, dwells[1:])
        step_time_v_vel = zip(steps / free_vels, free_vels)
        dwell_autocorr = zip(dwells[:-1], dwells[1:])
        step_autocorr = zip(steps[:-1], steps[1:])
        vels_autocorr = zip(free_vels[:-1], free_vels[1:])
    else:
        step_size_v_predwell = None
        step_size_v_postdwell = None
        step_vel_v_predwell = None
        step_vel_v_postdwell = None
        step_time_v_vel = None
        dwell_autocorr = None
        step_autocorr = None
        vels_autocorr = None
    return (dwells, free_vels, avg_free_vels, post_dwell_steps, post_free_vels,
            pre_dwell_steps, pre_free_vels, steps, vels, step_size_v_predwell,
            step_size_v_postdwell, step_vel_v_predwell, step_vel_v_postdwell,
            step_time_v_vel, dwell_autocorr, step_autocorr, vels_autocorr)


def main():
    # experiments = load_trajectories(['hcp', 'ccp', 'hcw', 'ccw'])
    # experiments = load_trajectories(['hfp', 'ffp', 'hfw', 'ffw'])
    experiments = load_trajectories(['hvp', 'vvp'])
    expt = 'hvp'
    steps_combined = list()
    pre_dwell_steps_combined = list()
    post_dwell_steps_combined = list()
    dwells_combined = list()
    vels_combined = list()
    free_vels_combined = list()
    avg_free_vels_combined = list()
    pre_free_vels_combined = list()
    post_free_vels_combined = list()
    step_predwell_combined = list()
    step_postdwell_combined = list()
    vel_predwell_combined = list()
    vel_postdwell_combined = list()
    step_time_vel_combined = list()
    dwell_autocorr_combined = list()
    step_autocorr_combined = list()
    vels_autocorr_combined = list()
    for trajectory in experiments[expt]:
        y_save, t_save = process_trajectory(trajectory, absolute_pause_threshold=1.)[1:]

        (dwells, free_vels, avg_free_vels, post_dwell_steps, post_free_vels,
         pre_dwell_steps, pre_free_vels, steps, vels, step_size_v_predwell,
         step_size_v_postdwell, step_vel_v_predwell, step_vel_v_postdwell,
         step_time_v_vel, dwell_autocorr, step_autocorr, vels_autocorr) = (
            extract_state_data(t_save, y_save))

        steps_combined.append(steps)
        dwells_combined.append(steps)
        vels_combined.append(vels)
        free_vels_combined.append(free_vels)
        avg_free_vels_combined.append(avg_free_vels)
        pre_dwell_steps_combined.append(pre_dwell_steps)
        post_dwell_steps_combined.append(post_dwell_steps)
        pre_free_vels_combined.append(pre_free_vels)
        post_free_vels_combined.append(post_free_vels)
        step_predwell_combined.append(step_size_v_predwell)
        step_postdwell_combined.append(step_size_v_postdwell)
        vel_predwell_combined.append(step_vel_v_predwell)
        vel_postdwell_combined.append(step_vel_v_postdwell)
        step_time_vel_combined.append(step_time_v_vel)
        dwell_autocorr_combined.append(dwell_autocorr)
        step_autocorr_combined.append(step_autocorr)
        vels_autocorr_combined.append(vels_autocorr)

    steps_combined = np.concatenate(steps_combined)
    dwells_combined = np.concatenate(dwells_combined)
    vels_combined = np.array(vels_combined)
    free_vels_combined = np.concatenate(free_vels_combined)
    avg_free_vels_combined = np.concatenate(avg_free_vels_combined)
    pre_dwell_steps_combined = np.array([i for i in pre_dwell_steps_combined
                                         if i is not None])
    post_dwell_steps_combined = np.array([i for i in post_dwell_steps_combined
                                         if i is not None])
    pre_free_vels_combined = np.array([i for i in pre_free_vels_combined
                                       if i is not None])
    post_free_vels_combined = np.array([i for i in post_free_vels_combined
                                        if i is not None])

    correlation_data = [
        step_predwell_combined, step_postdwell_combined, vel_predwell_combined,
        vel_postdwell_combined, step_time_vel_combined, dwell_autocorr_combined,
        step_autocorr_combined, vels_autocorr_combined
    ]
    processed_correlations = list()
    for combined_data in correlation_data:
        # Wild list comprehension to reshape autocorrelation data into an array
        processed_correlations.append(np.array([
            tup for sublist in combined_data if sublist is not None
            for tup in sublist
        ]))

    # print(np.sort(steps_combined))
    # print(np.sort(pre_dwell_steps_combined))
    # print(np.sort(post_dwell_steps_combined))
    # print(np.sort(dwells_combined))

    pre_step_times_combined = pre_dwell_steps_combined / pre_free_vels_combined
    int_step_times_combined = steps_combined / free_vels_combined
    post_step_times_combined = post_dwell_steps_combined / post_free_vels_combined

    plt.boxplot([steps_combined, pre_dwell_steps_combined, post_dwell_steps_combined],
                labels=['Steps between dwells', 'Steps entering domain',
                        'Steps leaving domain'])
    plt.ylabel('Step sizes $(\\mu m)$')
    plt.show()

    plt.boxplot([int_step_times_combined, pre_step_times_combined, post_step_times_combined],
                labels=['Steps between dwells', 'Steps entering domain',
                        'Steps leaving domain'])
    plt.ylabel('Step times (s)')
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

    plot_labels = [
        {'title': 'Step size vs Previous dwell', 'x_label': 'Step size (um)',
         'y_label': 'Previous dwell time (s)'},
        {'title': 'Step size vs Next dwell', 'x_label': 'Step size (um)',
         'y_label': 'Next dwell time (s)'},
        {'title': 'Step velocity vs Previous dwell',
         'x_label': 'Step velocity (um/s)',
         'y_label': 'Previous dwell time (s)'},
        {'title': 'Step velocity vs Next dwell',
         'x_label': 'Step velocity (um/s)', 'y_label': 'Next dwell time (s)'},
        {'title': 'Step time vs Step velocity', 'x_label': 'Step time (s)',
         'y_label': 'Step velocity (um/s)'},
        {'title': 'Dwell autocorrelation', 'x_label': 'First dwell time (s)',
         'y_label': 'Second dwell time (s)'},
        {'title': 'Step size autocorrelation',
         'x_label': 'First step size (um)', 'y_label': 'Second step size (um)'},
        {'title': 'Step velocity autocorrelation',
         'x_label': 'First velocity (um/s)',
         'y_label': 'Second velocity (um/s)'}
    ]

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    for i, data in enumerate(processed_correlations):
        ax = axs[i // 2, i % 2]
        try:
            ax.plot(data[:, 0], data[:, 1], 'o')
        except IndexError:
            pass
        ax.set_title(plot_labels[i]['title'])
        ax.set_xlabel(plot_labels[i]['x_label'])
        ax.set_ylabel(plot_labels[i]['y_label'])
        print(plot_labels[i]['title'])
        try:
            print(pearsonr(data[:, 0], data[:, 1]))
            print(spearmanr(data[:, 0], data[:, 1]))
        except IndexError:
            pass

    plt.tight_layout()
    plt.show()

    # plt.hist(pre_dwell_steps_combined)
    # plt.show()

    mu = np.mean(pre_step_times_combined)
    print('On rate before dwells is {}'.format(1/mu))
    x_plot = np.linspace(0, np.amax(pre_step_times_combined), num=200)
    plt.hist(pre_step_times_combined, density=True)
    plt.plot(x_plot, expon.pdf(x_plot, scale=mu))
    plt.title('Density of initial step times vs. a fitted exponential')
    plt.xlabel('Step time (s)')
    plt.ylabel('Probability density')
    plt.show()

    mu1 = np.mean(int_step_times_combined)
    print('On rate between dwells is {}'.format(1/mu1))
    x_plot1 = np.linspace(0, np.amax(int_step_times_combined), num=200)
    plt.hist(int_step_times_combined, density=True)
    plt.plot(x_plot1, expon.pdf(x_plot1, scale=mu1))
    plt.show()


def load_trajectories(keys):
    import os
    os.chdir(os.path.expanduser('~/thesis/vlado-data/'))

    experiments = dict()
    for key in keys:
        trajectory_files = glob.glob(key + '-t*.dat')
        trajectory_list = [np.loadtxt(f) for f in trajectory_files]
        experiments.update([(key, trajectory_list)])
    return experiments


if __name__ == '__main__':

    main()
