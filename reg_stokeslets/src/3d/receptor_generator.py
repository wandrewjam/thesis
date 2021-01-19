import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    a, b = 1.5, 0.5
    Nb = 6
    dh = b/Nb
    Na = int(a / b) * Nb

    # Generate parallelpiped
    node_list = []

    for patch in range(1, 7):
        if patch == 1 or patch == 2:
            i_list = np.arange(0, Na+1)
            # (1.5 - patch) gives 1/2 for patch 1, and -1/2 for patch 2
            x = np.ones(shape=(Na+1, Na+1)) * (1.5 - patch) * b
            y = i_list[:, None] * dh * np.ones(shape=(1, Na+1)) - a/2
            z = i_list[None, :] * dh * np.ones(shape=(Na+1, 1)) - a/2
        elif patch == 3 or patch == 4:
            i_list = np.arange(0, Na+1)
            j_list = np.arange(1, Nb)
            x = j_list[:, None] * dh * np.ones(shape=(1, Na+1)) - b/2
            y = i_list[None, :] * dh * np.ones(shape=(Nb-1, 1)) - a/2
            z = np.ones(shape=(Nb-1, Na+1)) * (3.5 - patch) * a
        elif patch == 5 or patch == 6:
            i_list = np.arange(1, Na)
            j_list = np.arange(1, Nb)
            x = j_list[:, None] * dh * np.ones(shape=(1, Na-1)) - b/2
            y = np.ones(shape=(Nb-1, Na-1)) * (5.5 - patch) * a
            z = i_list[None, :] * dh * np.ones(shape=(Nb-1, 1)) - a/2
        else:
            raise ValueError('patch is invalid')

        node_list.append(np.stack([x, y, z], axis=-1).reshape((-1, 3)))

    nodes = np.concatenate(node_list)

    D = np.array([1./4, 3./4, 3./4])
    norms = np.linalg.norm(nodes / D, axis=1)
    ellipse_nodes = nodes / norms[:, None]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ellipse_nodes[:, 0], ellipse_nodes[:, 1], ellipse_nodes[:, 2])
    ax.scatter([1, 1, 1, 1, -1, -1, -1, -1], [1, 1, -1, -1, 1, 1, -1, -1],
               [1, -1, 1, -1, 1, -1, 1, -1])
    plt.show()
