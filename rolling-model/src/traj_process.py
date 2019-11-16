import numpy as np


def main(filename):
    data = np.load(filename)
    vels = data['v_array']
    t = data['t_sample']

    traj = np.cumsum(vels[:,:-1] * (t[1:] - t[:-1]))


if __name__ == '__main__':
    import sys

    filename = sys.argv[1]
    main(filename)
