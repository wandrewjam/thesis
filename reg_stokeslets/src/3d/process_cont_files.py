import numpy as np
import os
from scipy.io import savemat


def main():
    data_dir = os.path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    os.chdir(data_dir)
    for entry in os.scandir(data_dir):
        if entry.name.endswith('.npz') and 'cont' not in entry.name:
            print(entry.name)
            cont_file = entry.name[:-4] + '_cont' + entry.name[-4:]
            try:
                with np.load(entry.path) as data:
                    t = data['t']
                    if t[-1] > 3.0001:
                        t *= .01

                    r_matrices = data['r_matrices']
                    if r_matrices.shape[-1] == 3:
                        r_matrices = r_matrices.transpose(1, 2, 0)

                    bond_array = data['bond_array']

                    if t[0] == 0. and np.abs(t[-1] - 3) < 1e-8:
                        os.remove(cont_file)
                    elif t[0] > 1e-12:
                        os.remove(entry.name)
                        os.remove(cont_file)
                    elif t[0] == 0. and t[-1] < 3.:
                        with np.load(cont_file) as data_cont:
                            if data_cont['t'][0] == 0.:
                                assert np.all(t == data_cont['t'][:len(t)])
                                os.replace(cont_file, entry.name)
                            elif t[0] == 0. and data_cont['t'][-1] == t[-1]:
                                os.remove(cont_file)
                            else:
                                assert t[-1] == data_cont['t'][0]

                                max_bonds = max([data_cont['bond_array'].shape[0], bond_array.shape[0]])
                                padded = np.pad(bond_array,
                                                pad_width=((0, max_bonds - bond_array.shape[0]),
                                                           (0, 0), (0, 0)),
                                                mode='constant', constant_values=-1)

                                cont_padded = np.pad(data_cont['bond_array'],
                                                     pad_width=((0, max_bonds - data_cont['bond_array'].shape[0]),
                                                                (0, 0), (0, 0)),
                                                     mode='constant', constant_values=-1)
                                new_t = np.concatenate((t[:-1], data_cont['t']))
                                new_x = np.concatenate((data['x'][:-1], data_cont['x']))
                                new_y = np.concatenate((data['y'][:-1], data_cont['y']))
                                new_z = np.concatenate((data['z'][:-1], data_cont['z']))
                                new_rmat = np.concatenate((r_matrices[..., :-1], data_cont['r_matrices']), axis=-1)
                                new_bond_array = np.concatenate((padded, cont_padded), axis=-1)

                                if 'draws' not in data.keys():
                                    continue
                                elif len(data['draws']) == 1 and 'draws' not in data_cont.keys():
                                    os.remove(entry.name)
                                    os.remove(cont_file)
                                    continue

                                new_draws = np.concatenate((data['draws'][:-1], data_cont['draws']))
                                assert len(new_draws) == len(new_t)

                                np.savez(data_dir + entry.name, t=new_t, x=new_x, y=new_y, z=new_z,
                                         r_matrices=new_rmat, bond_array=new_bond_array,
                                         receptors=data['receptors'], draws=new_draws)
                                savemat(data_dir + entry.name[:-4] + '.mat',
                                        {'t': new_t, 'x': new_x, 'y': new_y, 'z': new_z, 'R': new_rmat,
                                         'bond_array': new_bond_array, 'receptors': data['receptors'],
                                         'draws': new_draws})
            except FileNotFoundError:
                pass
        elif entry.name.endswith('_cont.npz'):
            base_filename = entry.name.replace('_cont', '')
            try:
                with np.load(entry.name) as data:
                    if data['t'][0] == 0.:
                        try:
                            with np.load(base_filename) as data_base:
                                if data_base['t'][-1] < data['t'][-1]:
                                    os.replace(entry.name, base_filename)
                                elif data_base['t'][-1] >= data['t'][-1]:
                                    os.remove(entry.name)
                        except FileNotFoundError:
                            os.replace(entry.name, base_filename)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    main()
