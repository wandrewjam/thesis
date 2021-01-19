import numpy as np
from scipy.io import savemat


if __name__ == '__main__':
    for expt_num in range(1, 9):
        data = np.load('data/bd_expt{:03d}.npz'.format(expt_num))
        savemat('data/bd_expt{:03d}'.format(expt_num),
                {'t': data['t'], 'x': data['x'], 'y': data['y'],
                 'z': data['z'], 'R': data['r_matrices'],
                 'bond_array': data['bond_array'],
                 'receptors': data['receptors']})

    print('Done!')
