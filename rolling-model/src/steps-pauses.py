import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(1010)
    steps = np.random.exponential(scale=0.1, size=10)
    pauses = np.random.exponential(scale=0.1, size=10)
    t = np.cumsum(pauses)
    displacement = np.cumsum(steps)

    plt.step(t, displacement, where='post')
    plt.xlabel('Time', fontsize=22)
    plt.ylabel('Displacement', fontsize=22)
    plt.tight_layout()
    plt.show()
