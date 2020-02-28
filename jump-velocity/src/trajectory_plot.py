import numpy as np
import matplotlib.pyplot as plt
from extract_trajectory_data import load_trajectories


if __name__ == '__main__':
    expts = {'Collagen PRP': ['hcp', 'ccp'],
             'Collagen Whole': ['hcw', 'ccw'],
             'Fibrinogen PRP': ['hfp', 'ffp'],
             'Fibrinogen Whole': ['hfw', 'ffw'],
             'vWF PRP': ['hvp', 'vvp']}

    for group, exp in expts.items():
        trajectories = load_trajectories(exp)
        for key, traj_list in trajectories.items():
            if key[0] == 'h':
                color = 'b'
            else:
                color = 'r'
            for traj in traj_list:
                plt.plot(traj[:, 0] - traj[0, 0], traj[:, 1], color,
                         linewidth=0.4)
        plt.title(group)
        plt.xlabel('Time Elapsed (s)')
        plt.ylabel('Distance Traveled ($\\mu m$)')
        plt.tight_layout()
        plt.show()
