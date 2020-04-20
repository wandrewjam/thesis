import numpy as np
import matplotlib.pyplot as plt
from extract_trajectory_data import load_trajectories, truncate_trajectory
from simulate import experiment
from free_velocity_processing import plot_dwells, plot_steps, get_free_velocities


if __name__ == '__main__':
    # Plot full trajectories
    expts = {'Collagen PRP': ['hcp', 'ccp'],
             'Collagen Whole': ['hcw', 'ccw'],
             'Fibrinogen PRP': ['hfp', 'ffp'],
             'Fibrinogen Whole': ['hfw', 'ffw', 'hfe', 'ffe'],
             'vWF PRP': ['hvp', 'vvp']}

    for group, exp in expts.items():
        trajectories = load_trajectories(exp)
        for key, traj_list in trajectories.items():
            if key[0] == 'h':
                color = 'b'
            else:
                color = 'r'
            if key == 'hfe':
                color = 'c'
            elif key == 'ffe':
                color = 'y'
            for traj in traj_list:
                plt.plot(traj[:, 0] - traj[0, 0], traj[:, 1], color,
                         linewidth=0.4)
        plt.title(group)
        plt.xlabel('Time Elapsed (s)')
        plt.ylabel('Distance Traveled ($\\mu m$)')
        plt.tight_layout()
        plt.show()

    # Plot truncated trajectories
    # expts = {'Collagen PRP': ['hcp', 'ccp'],
    #          'Collagen Whole': ['hcw', 'ccw'],
    #          'Fibrinogen PRP': ['hfp', 'ffp'],
    #          'Fibrinogen Whole': ['hfw', 'ffw'],
    #          'vWF PRP': ['hvp', 'vvp']}
    #
    # for group, exp in expts.items():
    #     trajectories = load_trajectories(exp)
    #     for key, traj_list in trajectories.items():
    #         if key[0] == 'h':
    #             color = 'b'
    #         else:
    #             color = 'r'
    #         for traj in traj_list:
    #             trunc_traj = truncate_trajectory(
    #                 traj, absolute_pause_threshold=0.)
    #             if trunc_traj.shape[0] > 0 and np.all(
    #                     trunc_traj[1:, 0] - trunc_traj[:-1, 0] > 0):
    #                 trunc_traj -= trunc_traj[0]
    #                 plt.plot(trunc_traj[:, 0], trunc_traj[:, 1], color,
    #                          linewidth=0.4)
    #
    #     plt.title(group)
    #     plt.xlabel('Time Elapsed (s)')
    #     plt.ylabel('Distance Traveled ($\\mu m$)')
    #     plt.tight_layout()
    #     plt.show()
