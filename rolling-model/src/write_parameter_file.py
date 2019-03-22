from default_values import set_parameters


def write_parameter_file(file_name, alg, correct_flux, M, N, time_steps, init,
                         trials, proc, **kwargs):
    """ Writes a file containing the parameters to run """
    (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history) = set_parameters(**kwargs)

    format_str = '{0:s} {1:}\n'

    with open(file_name+'.txt', 'w') as f:
        f.write(format_str.format('v_f', v_f))
        f.write(format_str.format('om_f', om_f))
        f.write(format_str.format('kappa', kappa))
        f.write(format_str.format('eta', eta))
        f.write(format_str.format('d', d))
        f.write(format_str.format('delta', delta))
        f.write(format_str.format('on', on))
        f.write(format_str.format('off', off))
        f.write(format_str.format('sat', sat))
        f.write(format_str.format('xi_v', xi_v))
        f.write(format_str.format('xi_om', xi_om))
        f.write(format_str.format('L', L))
        f.write(format_str.format('T', T))
        f.write(format_str.format('save_bond_history', save_bond_history))
        f.write(format_str.format('M', M))
        f.write(format_str.format('N', N))
        f.write(format_str.format('time_steps', time_steps))
        f.write(format_str.format('init', init))
        f.write(format_str.format('trials', trials))
        f.write(format_str.format('proc', proc))
        f.write(format_str.format('correct_flux', correct_flux))
        f.write(format_str.format('alg', alg))
        f.write(format_str.format('file_name', file_name))


def setup_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alg', choices=['bw', 'up'], default='up')
    parser.add_argument('-x', '--correct_flux', type=bool, default=False)
    parser.add_argument('-M', type=int, default=64)
    parser.add_argument('-N', type=int, default=64)
    parser.add_argument('-t', '--time_steps', type=int, default=10240 * 5)
    parser.add_argument('-i', '--init', default='free')
    parser.add_argument('-r', '--trials', type=int, default=100)
    parser.add_argument('-p', '--proc', type=int, default=4)
    parser.add_argument('-g', '--gamma', type=float)
    parser.add_argument('-k', '--kappa', type=float)
    parser.add_argument('-e', '--eta', type=float)
    parser.add_argument('-d', '--separation', type=float)
    parser.add_argument('-n', '--on', type=bool)
    parser.add_argument('-f', '--off', type=bool)
    parser.add_argument('-s', '--sat', type=bool)
    parser.add_argument('--xi_v', type=float)
    parser.add_argument('--xi_om', type=float)
    parser.add_argument('-L', type=float)
    parser.add_argument('-T', type=float)
    parser.add_argument('--bond_max', type=int)
    parser.add_argument('--save_bond_history', type=bool)
    parser.add_argument('file_name')

    return parser


if __name__ == '__main__':
    parser = setup_parser()

    args = parser.parse_args()
    args = vars(args)
    stripped_args = dict()
    for (key, val) in args.iteritems():
        if val is not None:
            stripped_args[key] = val

    write_parameter_file(**stripped_args)
    print('Wrote parameters to {}.txt'.format(stripped_args['file_name']))
