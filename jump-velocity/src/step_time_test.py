import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from simulate import experiment


if __name__ == '__main__':
    num_expts = 1024
    L = 100.
    a, eps1 = 0.14, 0.25 * 2.5 / L
    c, eps2 = .2, 20 * 2.5 / L

    results = [
        experiment(a / eps1, (1 - a) / eps1, c / eps2, (1 - c) / eps2)
        for _ in range(num_expts)
    ]

    trajectories = [np.stack([res[1], res[0], res[2]], axis=-1)
                    for res in results]

    step_time_list = []
    for traj in trajectories:
        t, y, state = traj.T

        # state[i] stores the platelet state in t[i]--t[i+1]
        bound = ((state == 2) + (state == 3))[:-1]
        bound = np.append([False], bound)
        # bound[i] stores the bound state in t[i-1]--t[i]

        trans = bound[1:] != bound[:-1]
        trans = np.append(trans, True)
        trans[0] = True
        y_save = y[trans]
        t_save = t[trans]

        assert len(y_save) % 2 == 0
        assert len(y_save) == len(t_save)

        if y_save.size == 2:
            continue

        del_t = t_save[1:] - t_save[:-1]

        step_times = del_t[::2]
        step_time_list.append(step_times)

    step_time_arr = np.concatenate(step_time_list)

    x_plot = np.sort(step_time_arr)
    y_plot = (np.arange(len(x_plot)) + 1.) / len(x_plot)

    x = np.linspace(0, x_plot[-1])

    plt.title('L = {}'.format(L))
    plt.step(x_plot, y_plot)
    plt.plot(x, expon.cdf(x, scale=eps2 / (1 - c)))
    plt.xlabel('Nondimensional time')
    plt.ylabel('Nondimensional distance')
    plt.legend(['Simulated step sizes', 'Predicted step sizes'])
    plt.tight_layout()
    plt.show()
