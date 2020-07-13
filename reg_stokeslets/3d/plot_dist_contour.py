import numpy as np
import matplotlib.pyplot as plt
import pickle


def spheroid_surface_area(a, b):
    assert a > b
    e = np.sqrt(1 - b**2 / a**2)
    return 2 * np.pi * a ** 2 * (1 + b ** 2 * np.arctanh(e) / (e * a ** 2))


def main():
    with open('dist_convergence.pkl') as f:
        conv = pickle.load(f)

    with open('dist_errs.pkl') as f:
        err = pickle.load(f)

    err_array = np.zeros(shape=(len(conv.keys()),
                                len(conv[conv.keys()[0]].keys())))

    distances = np.sort(conv.keys())
    n_array = np.sort(conv[distances[0]].keys())

    h = np.sqrt(spheroid_surface_area(1.5, .5) / (6 * n_array**2 + 1))
    epsilon = 0.6 * h

    for i, dist in enumerate(distances):
        exact = conv[dist][36]
        max_el = np.amax(np.abs(np.concatenate(exact, axis=1)))

        for j, n in enumerate(n_array):
            err_array[i, j] = err[dist][n] / max_el

    plt.contourf(distances, epsilon, err_array.T, [.01, .05, .1, .5, 1., 10.],
                 vmin=0., vmax=10.)
    plt.colorbar()
    plt.xlabel('Center of mass distance from wall')
    plt.ylabel('Regularization parameter $\\epsilon$')
    plt.show()


if __name__ == '__main__':
    main()
