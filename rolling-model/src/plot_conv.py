import numpy as np
import matplotlib.pyplot as plt

# Script to plot the results of the convergence analysis of the 3
# deterministic linear models

if __name__ == '__main__':
    in_str = raw_input('Enter v and om: ')
    in_str = in_str.split()
    v, om = float(in_str[0]), float(in_str[1])

    data = np.load('./data/conv_data/det_schemes_conv_v{:g}_om{:g}.npz'
                   .format(v, om))

    steps = data['steps']
    up_errs = data['up_errs']
    bw_errs = data['bw_errs']
    bn_errs = data['bn_errs']

    plt.loglog(steps, up_errs, 'k', label='Upwind scheme')
    plt.loglog(steps, bw_errs, 'c', label='Beam-Warming scheme')
    plt.loglog(steps, bn_errs, 'r', label='Semi-lagrangian scheme')
    plt.xlabel('$1/M$')
    plt.ylabel('$L^2$ errors')
    plt.xscale('log', basex=2)
    plt.legend()
    plt.show()
