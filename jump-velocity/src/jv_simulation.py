import numpy as np
import matplotlib.pyplot as plt


# def read_parameter_file(filename):
#     txt_dir = 'par-files/'
#     parlist = [('filename', filename)]
#
#     with open(txt_dir + filename + '.txt') as f:
#         while True:
#             command = f.readline().split()
#             if len(command) < 1:
#                 continue
#             if command[0] == 'done':
#                 break
#
#             key, value = command
#             if key == 'trials':
#                 parlist.append((key, int(value)))
#             else:
#                 parlist.append((key, float(value)))
#     return dict(parlist)


# def main(trials, eps1, eps2, a, c, filename):
#     assert np.minimum(eps1, eps2) < np.inf
#     if np.maximum(eps1, eps2) == np.inf:
#         # Two-state problem
#         if eps1 == np.inf:
#             # Define a, b to be the nonzero reaction rates
#             eps1, a = eps2, c
#         steps = np.random.exponential(scale=eps1/a, size=int(a*trials/eps1))
#         b = 1 - a
#         pauses = np.random.exponential(scale=eps1/b, size=int(a*trials/eps1)-1)
#         step_times = np.cumsum(steps)
#         while step_times[-1] < trials:
#             step_times = np.append(step_times, step_times[-1]
#                                    + np.random.exponential(eps1/a))
#             pauses = np.append(pauses, np.random.exponential(eps1/b))
#         indices = np.searchsorted(step_times, np.arange(stop=trials)+1)
#         step_split = np.split(steps, indices)
#         pause_split = np.split(pauses, indices-1)
#         for step in step_split:
#
#     b, d = 1 - a, 1 - c
#     ka, kb = a / (1 + eps1 / eps2), b / (1 + eps1 / eps2)
#     kc, kd = c / (1 + eps2 / eps1), d / (1 + eps2 / eps1)
#     coeff = 1/eps1 + 1/eps2
#
#     y, j = np.zeros(shape=1), np.zeros(shape=1, dtype='int')
#     t = np.zeros(shape=1)
#
#     while True:
#         if j[-1] == 0:
#             r_sum = np.cumsum(coeff * np.array([kb, kd]))
#         dt = np.random.exponential(r_sum[-1])


def experiment(rate_a, rate_b, rate_c, rate_d):
    assert np.minimum(rate_a, rate_b) > 0
    t = np.zeros(shape=1)
    y = np.zeros(shape=1)
    state = np.zeros(shape=1, dtype='int')
    if rate_c == 0:
        assert rate_d == 0
        while y[-1] < 1:
            if state[-1] == 0:
                dt = np.random.exponential(scale=1 / rate_b)
                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1] + dt)
                state = np.append(state, 1)
            elif state[-1] == 1:
                dt = np.random.exponential(scale=1 / rate_a)
                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1])
                state = np.append(state, 0)
            else:
                raise ValueError('State must be 0 or 1')
    else:
        assert rate_d > 0
        while y[-1] < 1:
            if state[-1] == 0:
                dt = np.random.exponential(scale=1 / (rate_b + rate_d))
                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1] + dt)
                r = np.random.rand(1)
                if r < rate_b / (rate_b + rate_d):
                    state = np.append(state, 1)
                else:
                    state = np.append(state, 2)
            elif state[-1] == 1:
                dt = np.random.exponential(scale=1 / (rate_a + rate_d))
                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1])
                r = np.random.rand(1)
                if r < rate_a / (rate_a + rate_d):
                    state = np.append(state, 0)
                else:
                    state = np.append(state, 3)
            elif state[-1] == 2:
                dt = np.random.exponential(scale=1 / (rate_b + rate_c))
                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1])
                r = np.random.rand(1)
                if r < rate_c / (rate_b + rate_c):
                    state = np.append(state, 0)
                else:
                    state = np.append(state, 3)
            elif state[-1] == 3:
                dt = np.random.exponential(scale=1 / (rate_a + rate_c))
                t = np.append(t, t[-1] + dt)
                y = np.append(y, y[-1])
                r = np.random.rand(1)
                if r < rate_a / (rate_a + rate_c):
                    state = np.append(state, 2)
                else:
                    state = np.append(state, 1)
            else:
                raise ValueError('State must be one of 0, 1, 2, or 3')
    excess = y[-1] - 1
    y[-1] -= excess
    t[-1] -= excess
    return 1 / t[-1]


def main():
    a, c = .5, .2
    b, d = 1 - a, 1 - c
    eps1, eps2 = .1, 1

    rate_a, rate_b = a / eps1, b / eps1
    rate_c, rate_d = c / eps2, d / eps2

    num_expt = 1000
    vels = list()
    for i in range(num_expt):
        vels.append(experiment(rate_a, rate_b, rate_c, rate_d))

    plt.hist(vels, density=True)
    plt.show()

    x = np.sort(vels)
    x = np.insert(x, 0, 0)
    y = np.linspace(0, 1, len(x))
    x = np.append(x, 1.5)
    y = np.append(y, 1)
    plt.step(x, y, where='post')
    plt.show()


if __name__ == '__main__':
    # import sys
    # filename = sys.argv[1]
    #
    # pars = read_parameter_file(filename)
    # main(**pars)
    main()
