import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


def read_parameter_file(filename):
    txt_dir = 'par-files/'
    parlist = [('filename', filename)]

    with open(txt_dir + filename + '.txt') as f:
        while True:
            command = f.readline().split()
            if len(command) < 1:
                continue
            if command[0] == 'done':
                break

            key, value = command
            if key == 'N' or key == 's_samp':
                parlist.append((key, int(value)))
            elif key == 'scheme':
                parlist.append((key, value))
            else:
                parlist.append((key, float(value)))
    return dict(parlist)


def main(filename):
    data = np.load(filename + '.npz')
    pars = read_parameter_file(filename + '.txt')
    y, s_eval = data['y'], data['s_eval']
    u0_bdy, u1_bdy = data['u0_bdy'], data['u1_bdy']

    eps1, eps2 = pars['eps1'], pars['eps2']
    b, d = 1 - pars['a'], 1 - pars['c']

    plt.plot(s_eval, u1_bdy, s_eval, u0_bdy, s_eval, cumtrapz(u1_bdy, s_eval, initial=0))
    plt.axhline(1 - np.exp(-(1/eps1 + 1/eps2) * (b / (1 + eps1/eps2) + d / (1 + eps2/eps1))), color='k')
    plt.show()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]

    main(filename)
