import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    conv_data = np.loadtxt('convergence_test.dat')
    eps_data = np.loadtxt('epsilon_refinement.dat')

    plt.plot(conv_data[:, 0], conv_data[:, 1] / (4 * np.pi), 'b.-')
    plt.title('Regularization, $\\epsilon = 0.01$')
    plt.xlabel('Grid size ($\\Delta s = \\pi / 2N$)')
    plt.ylabel('$|| u_3 + 1 ||_2$')
    plt.show()

    plt.plot(eps_data[:, 1], eps_data[:, 0] / (4 * np.pi), 'b.-')
    plt.title('Grid size 6x24x24')
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$||u_3 + 1 ||_2$')
    plt.show()
