import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.special import gammainc
from step_fit_modified import fit_trunc_gamma


if __name__ == '__main__':
    data = gamma.rvs(a=2, scale=0.5, size=2 ** 14)
    data = data[data < 2]

    k, theta = fit_trunc_gamma(data)
    s = np.linspace(0, 2)
    plt.hist(data, density=True)
    plt.plot(s, gamma.pdf(s, a=k, scale=theta) / (gammainc(k, 2 / theta)))
    plt.show()
    print(k, theta)
