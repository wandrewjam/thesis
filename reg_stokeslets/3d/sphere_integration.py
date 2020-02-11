import numpy as np
from scipy.integrate import dblquad
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cProfile
from timeit import default_timer as timer


def lagrange_map(x_coords):
    x, y, z = x_coords
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    return np.stack([theta, phi])


def cart_map(s_coords):
    theta, phi = s_coords
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z])


def area(corners):
    # Each column is a point
    angles = list()
    normals = list()
    n = corners.shape[1]
    corners_copy = np.tile(corners, (1, 2))
    for i in range(n):
        normals.append(np.cross(corners_copy[:, i], corners_copy[:, i+1]))
    normals_copy = np.tile(normals, (2, 1)).T
    for i in range(n):
        angles.append(np.arccos(-np.dot(normals_copy[:, i-1], normals_copy[:, i]) /
                                (np.linalg.norm(normals_copy[:, i-1])
                                 * np.linalg.norm(normals_copy[:, i]))))
    return np.abs(np.sum(angles)) - (n - 2)*np.pi


def f(lag_coords, s0, i):
    eps = 0.01
    cart_coords = cart_map(lag_coords)
    x0 = cart_map(s0)
    output = list()
    for k in range(cart_coords.shape[1]):
        x = cart_coords[:, k]
        del_x = x - x0
        r2 = np.dot(del_x, del_x)

        stokeslet = (np.eye(3) * (r2 + 2*eps**2) / ((r2 + eps**2)**(3/2))
                     + np.outer(del_x, del_x) / ((r2 + eps**2)**(3/2)))
        force = -3. / 2 * np.array([0, 0, 1])
        output.append(np.dot(stokeslet, force)[i])
    return np.array(output)


def main():
    start = timer()
    N = 8
    i, j, k = [np.arange(N + 1) for _ in range(3)]
    ind_nodes = np.array([[ci, cj, ck] for ci in i for cj in j for ck in k]).T
    cart_nodes = 2. / N * ind_nodes - 1
    boundary_nodes = np.nonzero(np.linalg.norm(cart_nodes, ord=np.inf, axis=0) == 1)
    cart_nodes = cart_nodes[:, boundary_nodes[0]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf_nodes = cart_map(lagrange_map(cart_nodes))
    ax.scatter(surf_nodes[0], surf_nodes[1], surf_nodes[2])
    plt.show()

    weights = np.ones(shape=cart_nodes.shape[1])
    weights = np.random.uniform(0, 1, size=cart_nodes.shape[1])
    # weights[np.sum(np.abs(cart_nodes), axis=0) == 3] = 4./3
    ind_nodes = ind_nodes[:, boundary_nodes[0]]
    patch_offsets = [np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
                     np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
                     np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])]
    patches = list()
    for k in range(ind_nodes.shape[1]):
        current_node = ind_nodes[:, k]
        for offset in patch_offsets:
            diffs = (offset + current_node).T[:, None, :] - ind_nodes[:, :, None]
            norms = np.linalg.norm(diffs, axis=0)
            indices = np.nonzero(norms == 0)
            if len(indices[1]) == 4:
                patches.append(indices[0][np.argsort(indices[1])])
    patches = np.array(patches)
    lag_nodes = lagrange_map(cart_nodes)
    area(cart_nodes[:, patches[0]])
    lagrange_map(cart_nodes[:, patches[0]])
    areas = [area(cart_nodes[:, patch]) for patch in patches]
    integrand_values = [[
        f(lagrange_map(cart_nodes), lag_nodes[:, k], j) / (8 * np.pi)
        for j in range(3)] for k in range(lag_nodes.shape[1])]
    accum = [[0, 0, 0] for _ in range(lag_nodes.shape[1])]
    for k in range(lag_nodes.shape[1]):
        for i, patch in enumerate(patches):
            for j in range(3):
                accum[k][j] += np.mean(integrand_values[k][j][patch] * weights[patch]) * areas[i]
    accum = np.array(accum)
    err = [0, 0, 0]
    true = [0, 0, -1]
    for i, patch in enumerate(patches):
        for j in range(3):
            err[j] += np.mean((accum[patch, j] - true[j]) ** 2 * weights[patch]) * areas[i]
    end = timer()
    print(err)
    print(end - start)


if __name__ == '__main__':
    cProfile.run('main()')
