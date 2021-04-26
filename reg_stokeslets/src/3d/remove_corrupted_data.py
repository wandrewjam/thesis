import numpy as np
import os


def main():
    data_dir = os.path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    os.chdir(data_dir)

    for entry in os.scandir():
        if entry.name.endswith('.npz'):
            try:
                with np.load(entry.name) as data:
                    if data['t'].shape[0] != data['bond_array'].shape[-1]:
                        os.remove(entry.name)
            except KeyError:
                pass


if __name__ == '__main__':
    main()
