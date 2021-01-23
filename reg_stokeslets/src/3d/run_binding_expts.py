import numpy as np
from binding_expt import parse_file, save_info


def main(num_expts, filename, runner, random_initial='F'):
    import os
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
    pars = parse_file(filename)

    all_filenames = []

    for i in range(num_expts):
        file_i = [int(f[6:9]) for f in os.listdir(txt_dir) 
                  if (f[:6] == 'bd_run' and f[6:9] != 'ner')]
        try:
            k = max(file_i)
        except ValueError:
            k = -1
        filename = 'bd_run{:03d}'.format(k+1)
        all_filenames.append(filename + '\n')
        seed = np.random.randint(2**32)
        pars.update([('seed', seed), ('filename', filename)])
        if random_initial == 'T':
            height = np.random.uniform(0.7, 1.4)
            theta  = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(0, np.pi)
            cp, sp = np.cos(phi), np.sin(phi)
            ct, st = np.cos(theta), np.sin(theta)
            e_m = np.array([cp, ct * sp, st * sp])
            pars['x1'] = height
            pars['emx'], pars['emy'], pars['emz'] = e_m

        pars['order'] = '2nd'
        save_info(**pars)

    with open(txt_dir + runner + '.txt', 'w') as f:
        f.writelines(all_filenames)


if __name__ == '__main__':
    import sys

    try:
        main(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
    except IndexError:
        main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
