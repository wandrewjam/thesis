import matplotlib.pyplot as plt
from matplotlib import animation
from jv import delta_h
import numpy as np


def main(filename, show_ani=False):
    def animate(i):
        line_u0.set_ydata(np.append(delta_h(-s_eval[t_store[i]], h), u0_data[:, i]))
        line_u1.set_ydata(u1_data[:, i])
        line_v.set_ydata(v_data[:, i])
        line_f.set_ydata(f_data[:, i])
        line_vf.set_ydata(vf_data[:, i])
        vline.set_xdata([s_eval[t_store[i]]] * 2)
        return line_u0, line_u1, line_v, line_f, line_vf, vline

    npz_dir = '/Users/andrewwork/thesis/jump-velocity/npz-files/'
    plot_dir = '/Users/andrewwork/thesis/jump-velocity/plots/'
    data = np.load(npz_dir + filename + '.npz')
    y_store, t_store = data['y_store'], data['t_store']
    y, s_eval = data['y'], data['s_eval']
    u0_data, u1_data, v_data, f_data, vf_data = data['u0_data'], data['u1_data'], data['v_data'], data['f_data'], data['vf_data']

    h = y[1] - y[0]
    fig, ax = plt.subplots()
    line_u0, line_u1, line_v, line_f, line_vf = ax.plot(
        y[np.append(0, y_store + 1)], np.append(delta_h(s_eval[0], h), u0_data[:, 0]), y[y_store], u1_data[:, 0],
        y[y_store], v_data[:, 0],
        y[y_store], f_data[:, 0], y[y_store], vf_data[:, 0])
    line_u0.set_label('$q_U^0$')
    line_u1.set_label('$q_U^1$')
    line_v.set_label('$q_V$')
    line_f.set_label('$q_F$')
    line_vf.set_label('$q_{VF}$')
    ax.set_ylim(bottom=-1, top=11)
    vline = ax.axvline(s_eval[0], color='k')
    vline.set_label('$y = t$')
    ax.legend(loc='upper right')
    ax.set_xlabel('$y$')
    ax.set_ylabel('Probability density')
    ani = animation.FuncAnimation(
        fig, animate, frames=t_store.shape[0], interval=25)
    ani.save(plot_dir + filename + '.mp4')
    if show_ani:
        plt.show()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]

    main(filename, show_ani=True)
