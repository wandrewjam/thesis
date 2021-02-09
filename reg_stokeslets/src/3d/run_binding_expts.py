import numpy as np
from binding_expt import parse_file, save_info
from motion_integration import find_min_separation


def main(num_expts, filename, runner, random_initial='F', order='2nd'):
    import os
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
    pars = parse_file(filename)

    all_filenames = []

    for i in range(num_expts):
        file_i = []

        for f in os.listdir(txt_dir):
            if f[:6] == 'bd_run' and f[6:9] != 'ner':
                try:
                    file_i.append(int(f[6:10]))
                except ValueError:
                    file_i.append(int(f[6:9]))

        # try:
        #     file_i = [int(f[6:10]) for f in os.listdir(txt_dir)
        #               if (f[:6] == 'bd_run' and f[6:9] != 'ner')]
        # except ValueError:
        #     file_i = [int(f[6:10]) for f in os.listdir(txt_dir)
        #               if (f[:6] == 'bd_run' and f[6:9] != 'ner')]

        try:
            k = max(file_i)
        except ValueError:
            k = -1
        filename = 'bd_run{:06d}'.format(k+1)
        all_filenames.append(filename + '\n')
        seed = np.random.randint(2**32)
        pars.update([('seed', seed), ('filename', filename)])
        if random_initial == 'T':
            while True:
                height = np.random.uniform(0.7, 1.4)
                theta  = np.random.uniform(-np.pi, np.pi)
                phi = np.random.uniform(0, np.pi)

                cp, sp = np.cos(phi), np.sin(phi)
                ct, st = np.cos(theta), np.sin(theta)
                e_m = np.array([cp, ct * sp, st * sp])
                sep = find_min_separation(height, e_m)
                
                if sep > 0:
                    break

            pars['x1'] = height
            pars['emx'], pars['emy'], pars['emz'] = e_m

        pars['order'] = 'radau'
        save_info(**pars)
        
    with open(txt_dir + runner + '.txt', 'w') as f:
        f.writelines(all_filenames)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('num_expts', type=int)
    parser.add_argument('filename')
    parser.add_argument('runner')
    parser.add_argument('-r', '--randomize', action='store_true')
    parser.add_argument('--order', default='2nd',
                        choices=['radau', '1st', '2nd', '4th'])

    args = parser.parse_args()

    main(args.num_expts, args.filename, args.runner, args.randomize, args.order)
    # try:
    #     main(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
    # except IndexError:
    #     main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
