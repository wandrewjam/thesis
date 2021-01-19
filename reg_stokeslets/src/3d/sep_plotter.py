import numpy as np
import matplotlib.pyplot as plt
from motion_integration import n_picker


def main():
    sep = np.linspace(.04, .07)
    ns = np.zeros(shape=sep.shape)
    for i in range(len(sep)):
        ns[i] = n_picker(sep[i])

    plt.plot(sep, ns)
    plt.xlabel('Platelet separation distance')
    plt.ylabel('N chosen')
    plt.show()


if __name__ == '__main__':
    main()
