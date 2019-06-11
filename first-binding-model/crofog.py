import numpy as np


def crofog(y):
    Dl, Dh = -1e-9, 2e-6

    return (np.arctan(19 - np.abs(25 - y))/np.pi + .5)*(Dh - Dl) + Dl + (np.arctan(6.6 - 1.2*np.abs(12 - y))/np.pi + .5)*2*Dh/5 + (np.arctan(6.6 - 1.2*np.abs(38 - y))/np.pi + .5)*2*Dh/5


if __name__ == '__main__':
    print('Minimum diffusion is {:g}'.format(crofog(0)))
    print('Midline diffusion is {:g}'.format(crofog(25)))

    import matplotlib.pyplot as plt
    y = np.linspace(0, 50, num=500)
    plt.plot(y, crofog(y))
    plt.show()
