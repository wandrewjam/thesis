import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    scale, trials = 1, 2**15
    exp = np.random.exponential(scale=scale, size=(trials, 2))

    sum_exp = np.sum(exp, axis=1)

    z = np.linspace(0, np.amax(sum_exp), num=2**9)
    rate = 1/scale
    plt.hist(sum_exp, density=True)
    plt.plot(z, rate**2*z*np.exp(-rate*z))
    plt.show()

    plt.step(np.sort(sum_exp), (np.arange(sum_exp.size)+1.)/sum_exp.size,
             where='post')
    plt.plot(z, 1 - (1 + rate*z)*np.exp(-rate*z))
    plt.show()
