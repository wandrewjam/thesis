import numpy as np
from scipy.io import savemat


def main(plt_num):
    data = np.load('data/fine{}.npz'.format(plt_num))
    savemat('data/fine{}.mat'.format(plt_num), data)


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]))
