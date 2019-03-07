def write_parameter_file(file_name, alg, correct_flux, M, N, time_steps, init,
                         trials, proc, **kwargs):
    """ Writes a file containing the parameters to run """
    gamma = kwargs.pop('gamma')
    (kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history) = set_parameters(**kwargs)[2:]

    format_str = '{0:s} {1:}\n'

    with open(file_name+'.txt', 'w') as f:
        f.write(format_str.format('gamma', gamma))
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


if __name__ == '__main__':
    import sys
    file_name = sys.argv[1]
    write_parameter_file(file_name)
