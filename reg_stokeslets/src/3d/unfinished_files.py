import numpy as np
import os
import sys


def main(count):
    data_dir = os.path.expanduser('~/thesis/reg_stokeslets/data/bd_run')
    os.chdir(data_dir)
    i = 0
    for entry in os.scandir(data_dir):
        if entry.name.endswith('.npz') and '_cont' not in entry.name:
            with np.load(entry.name) as data:
                if data['t'][-1] < 3 - 1e-12 and 'draws' in data.keys():
                    i += 1
                    if 8 * (count - 1) <= i < 8 * count:
                        print(entry.name[:-4])


if __name__ == '__main__':
    run_count = int(sys.argv[1])
    main(run_count)
