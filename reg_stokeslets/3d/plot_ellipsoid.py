import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D


def define_parametric_vars():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_wall = np.zeros(shape=(2, 2))
    y_wall = np.array([[-2, 2],
                       [-2, 2]])
    z_wall = np.array([[-2, -2],
                       [2, 2]])
    xp = .5 * np.outer(np.ones(np.size(u)), np.cos(v))
    yp = 1.5 * np.outer(np.cos(u), np.sin(v))
    zp = 1.5 * np.outer(np.sin(u), np.sin(v))
    return x_wall, xp, y_wall, yp, z_wall, zp


def generate_3d_plot(em):
    x_wall, xp, y_wall, yp, z_wall, zp = define_parametric_vars()
    ph = np.arccos(em[0])
    th = np.arctan2(em[2], em[1])
    R = np.array([
        [np.cos(ph), -np.sin(ph), 0],
        [np.cos(th) * np.sin(ph), np.cos(th) * np.cos(ph), -np.sin(th)],
        [np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), np.cos(th)]
    ])

    p_array = np.stack([xp, yp, zp], axis=1)
    x, y, z = np.dot(R, p_array)
    x += 1.2

    x_vec = np.linspace(0, np.amax(x), 5)[1:, None] * np.ones(shape=(1, 5))
    y_vec = np.linspace(-2, 2, 5)[None, :] * np.ones(shape=(4, 1))
    z_vec = -2 * np.ones(shape=(4, 5))
    w_vec = x_vec
    u_vec = np.zeros(shape=w_vec.shape)
    v_vec = np.zeros(shape=w_vec.shape)

    # Output to Matlab data, need x, y, z, and x_wall, y_wall, z_wall
    mat_dict = {
        'x': x, 'y': y, 'z': z,
        'x_wall': x_wall, 'y_wall': y_wall, 'z_wall': z_wall,
        'x_vec': x_vec, 'y_vec': y_vec, 'z_vec': z_vec,
        'u_vec': u_vec, 'v_vec': v_vec, 'w_vec': w_vec
    }
    savemat('data/e1_data', mat_dict)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='b')
    ax.plot_surface(x_wall, y_wall, z_wall, color='gray')
    ax.quiver3D(x_vec, y_vec, z_vec, u_vec, v_vec, w_vec, length=.3)

    plt.show()


def main():
    data = np.load('data/fine48.npz')
    em_arr = np.stack([data[key] for key in ['e1', 'e2', 'e3']], axis=0)
    t = data['t']
    imax = 280

    i0 = 0
    i1 = np.argmin(em_arr[2, :imax])
    i2 = np.argmax(em_arr[0, :imax])
    i3 = np.argmax(em_arr[2, :imax])
    i4 = np.argmin(em_arr[0, 1:imax]) + 1

    for i in [i0, i1, i2, i3, i4]:
        em = em_arr[:, i]
        generate_3d_plot(em)
        print('t = {}'.format(t[i]))


if __name__ == '__main__':
    main()
