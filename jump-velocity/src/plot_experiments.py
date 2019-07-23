import numpy as np
from matplotlib import pyplot as plt


def plot_experiments(vels, x_plot, r_pdf, r_cdf, f_pdf, f_cdf):
    """Plots histogram and ECDF of average velocity data

    Parameters
    ----------
    vels : array_like
        Array of average rolling velocities
    x_plot
    r_pdf
    r_cdf
    f_pdf
    f_cdf

    Returns
    -------
    None
    """

    density_fig, density_ax = plt.subplots()
    density_ax.hist(vels, density=True, label='Sample data')

    density_ax.plot(x_plot, r_pdf, label='Adiabatic reduction')
    density_ax.plot(x_plot, f_pdf, label='Full model')
    density_ax.legend(loc='best')
    plt.show()

    x_cdf = np.sort(vels)
    x_cdf = np.insert(x_cdf, 0, 0)
    y_cdf = np.linspace(0, 1, len(x_cdf))
    x_cdf = np.append(x_cdf, 1.5)
    y_cdf = np.append(y_cdf, 1)

    cdf_fig, cdf_ax = plt.subplots()
    cdf_ax.step(x_cdf, y_cdf, where='post', label='Sample data')
    cdf_ax.plot(x_plot, r_cdf, label='Adiabatic reduction')
    cdf_ax.plot(x_plot, f_cdf, label='Full model')
    cdf_ax.legend(loc='best')
    plt.show()


def main(filename):
    sim_dir = 'dat-files/simulations/'
    est_dir = 'dat-files/ml-estimates/'

    vels = np.loadtxt(sim_dir + filename + '-sim.dat')
    mles = np.loadtxt(est_dir + filename + '-est.dat')

    x_plot, r_pdf, r_cdf, f_pdf, f_cdf = [mles[:, i] for i in range(5)]
    plot_experiments(vels, x_plot, r_pdf, r_cdf, f_pdf, f_cdf)


if __name__ == '__main__':
    import os
    import sys
    filename = sys.argv[1]
    os.chdir(os.path.expanduser('~/thesis/jump-velocity'))

    main(filename)
