import numpy as np
from binding_expt import parse_file, save_info


def main(num_expts, filename, runner):
    import os
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/3d/par-files/')
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
        save_info(**pars)

    with open(txt_dir + runner + '.txt', 'w') as f:
        f.writelines(all_filenames)


if __name__ == '__main__':
    import sys

    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
