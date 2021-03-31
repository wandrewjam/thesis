import numpy as np
import pickle as pkl
import os
from binding_expt import parse_file


def main():
    dir_path = os.path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    for entry in os.scandir(dir_path):
        if entry.path.endswith('070.pkl'):
            process_pkl_file(dir_path, entry)


def process_pkl_file(dir_path, entry):
    try:
        with open(entry.path, 'rb') as f:
            rng_history = pkl.load(f)
    except UnicodeDecodeError as e:
        with open(entry.path, 'rb') as f:
            rng_history = pkl.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', entry.name, ':', e)
        raise

    filename = entry.name[:-4]
    draws = extract_draws(filename, rng_history)
    save_new_npz(dir_path, draws, filename)
    # os.remove(entry.path)


def save_new_npz(dir_path, draws, filename):
    with np.load(dir_path + filename + '.npz') as data:
        save_dict = {}
        for key, value in data.items():
            if key[:3] != 'arr':
                save_dict[key] = value

        save_dict['draws'] = draws
        np.savez(dir_path + filename + '.npz', **save_dict)


def extract_draws(filename, rng_history):
    pars = parse_file(filename)
    rng = np.random.RandomState(pars['seed'])
    draws = []
    for j in range(len(rng_history)):
        for i in range(0, 10**6):
            if (np.all(rng.get_state()[1] == rng_history[j][1])
                    and rng.get_state()[2] == rng_history[j][2]):
                break
            rng.rand()
        # while i < 10 ** 6 and (
        #         np.all(rng.get_state()[1] != rng_history[j][1])
        #         or rng.get_state()[2] != rng_history[j][2]
        # ):
        #
        #     i += 1
        assert i < 10 ** 6
        draws.append(i)
    return draws


if __name__ == '__main__':
    main()
