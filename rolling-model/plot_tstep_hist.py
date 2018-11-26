# Script for plotting the timestep histogram

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = raw_input('Filename to load: ')
    data = np.load('./data/mov_rxns/'+filename)
    data = [data[key] for key in data]

    fig, ax = plt.subplots()

    ax.hist(np.hstack(data), bins=10)
    plt.show()

