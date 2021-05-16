import numpy as np
from scipy.stats import beta
from binding_expt import parse_file, save_info
from motion_integration import find_min_separation


def pick_random_height():
    a = 1.5
    b = (a - 1) / 0.043 - a + 2
    loc = 0.55
    scale = 3.45

    while True:
        r = beta.rvs(a=a, b=b, loc=loc, scale=scale)
        assert 0.55 < r < 4.

        if r < 1.65:
            break

    return r


def main(num_expts, runner, random_initial=False, catch_slip=False,
         start_numbering=-1, **pars):
    import os
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
    defaults = parse_file('default')

    all_filenames = []

    if random_initial:
        assert num_expts <= 512
        saved_positions = np.load(os.path.expanduser(
            '~/thesis/reg_stokeslets/src/3d/r_pos.npy'))

    for i in range(num_expts):
        par_dict = defaults.copy()
        assert type(start_numbering) is int
        if start_numbering == -1:
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
        else:
            assert start_numbering > -1

            k = start_numbering + i - 1
        
        runner_num = runner[-4:]
        filename = 'bd_run{}{:03d}'.format(runner_num, k+1)
        all_filenames.append(filename + '\n')
        seed = np.random.randint(2**32)

        par_dict.update([
            ('seed', seed), ('filename', filename)])
        par_dict.update(pars)

        if catch_slip:
            par_dict.update([('dimk0_on2', 5.)])
        else:
            par_dict.update([('dimk0_on2', 0.)])

            # height = pick_random_height()
            # while True:
            #     theta = np.random.uniform(-np.pi, np.pi)
            #     phi = np.random.uniform(0, np.pi)
            #
            #     cp, sp = np.cos(phi), np.sin(phi)
            #     ct, st = np.cos(theta), np.sin(theta)
            #     e_m = np.array([cp, ct * sp, st * sp])
            #     sep = find_min_separation(height, e_m)
            #
            #     if sep > 0.0154:
            #         break

        if random_initial:
            height = saved_positions[i+128, 0]
            e_m = saved_positions[i+128, 1:4]
            par_dict['x1'] = height
            par_dict['emx'], par_dict['emy'], par_dict['emz'] = e_m

        filename = par_dict.pop('filename')
        save_info(filename, **par_dict)
        
    with open(txt_dir + runner + '.txt', 'w') as f:
        f.writelines(all_filenames)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('num_expts', type=int)
    # parser.add_argument('filename')
    parser.add_argument('runner')
    parser.add_argument('-r', '--randomize', action='store_true')
    parser.add_argument('--order', default='2nd',
                        choices=['radau', '1st', '2nd', '4th'])
    parser.add_argument('-s', '--start_numbering', default=-1, type=int)
    parser.add_argument('-l', '--rest_length', default=0.1, type=float)

    parser.add_argument('-n', '--k_on', default=5.0, type=float)
    parser.add_argument('-f', '--k_off', default=5.0, type=float)
    parser.add_argument('-m', '--receptor_multiplier', default=1, type=int)
    parser.add_argument('-c', '--catch_slip', action='store_true')

    args = parser.parse_args()

    main(args.num_expts, args.runner, args.randomize,
         start_numbering=args.start_numbering,
         receptor_multiplier=args.receptor_multiplier, order=args.order,
         l_sep1=args.rest_length, dimk0_on=args.k_on, dimk0_off=args.k_off,
         catch_slip=args.catch_slip)

    # try:
    #     main(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
    # except IndexError:
    #     main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
