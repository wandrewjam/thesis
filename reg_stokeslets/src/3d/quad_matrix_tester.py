import numpy as np
import matplotlib.pyplot as plt
from motion_integration import integrate_motion, evaluate_motion_equations


def main():
    arr_keys = ['x1', 'x2', 'x3', 'e1', 'e2', 'e3']
    fine72 = np.load('data/fine72.npz')
    coarse72 = np.load('data/coarse72.npz')
    fine74 = np.load('data/fine74.npz')
    coarse74 = np.load('data/coarse74.npz')

    time1 = 0.
    ind1 = 0
    pos1 = [fine72[key][ind1] for key in arr_keys]
    pos1 = np.array(pos1)

    time2 = 11.
    ind2 = np.nonzero(fine72['t'] == 11.)[0][0]
    pos2 = [fine72[key][ind2] for key in arr_keys]
    pos2 = np.array(pos2)

    time3 = 20.
    ind3 = np.nonzero(fine72['t'] == 20.)[0][0]
    pos3 = [fine72[key][ind3] for key in arr_keys]
    pos3 = np.array(pos3)

    time4 = 0.
    ind4 = 0
    pos4 = [fine74[key][ind4] for key in arr_keys]
    pos4 = np.array(pos4)

    time5 = 55.
    ind5 = np.nonzero(fine74['t'] == 55.)[0][0]
    pos5 = [fine74[key][ind5] for key in arr_keys]
    pos5 = np.array(pos5)

    def exact_vels(em):
        return np.zeros(6)

    t_interval = 0.25
    t_steps = 2
    evaluate_motion_equations.counter = 0

    s_dict = {}

    for n_nodes in [8, 16]:
        s_matrices = []
        for start, init in zip([time1, time2, time3, time4, time5],
                               [pos1, pos2, pos3, pos4, pos5]):
            s_matrices.append(
                integrate_motion([start, start + t_interval], t_steps, init, exact_vels, n_nodes, a=1.5, b=0.5,
                                 domain='wall', order='2nd', adaptive=False, save_quad_matrix_info=True)[-1]
            )

        s_dict.update([(n_nodes, s_matrices)])

    cond_dict = {}
    for n_nodes, expt_list in s_dict.items():
        for i, run in enumerate(expt_list):
            for j, time_step in enumerate(run):
                for k, eval in enumerate(time_step):
                    # Do I also want to try a diagonal preconditioner?
                    D = np.diag(eval)
                    cond_dict.update([((n_nodes, i, j, k),
                                       (np.linalg.cond(eval),
                                        np.linalg.cond(eval / D[:, None])))])

    cond_array = np.zeros(shape=(len(cond_dict), 6))
    for i, key in enumerate(cond_dict.keys()):
        cond_array[i, :4] = key
        cond_array[i, 4:] = cond_dict[key]

    svd_dict = {}
    for n_nodes, expt_list in s_dict.items():
        for i, run in enumerate(expt_list):
            for j, time_step in enumerate(run):
                diff = time_step[1] - time_step[0]
                svd = np.abs(np.linalg.eigvalsh(diff, UPLO='U'))
                svd_dict.update([((n_nodes, i, j), svd)])

    svd_array = np.zeros(shape=(len(svd_dict), 4617))
    for i, key in enumerate(svd_dict.keys()):
        svd_array[i, :3] = key
        svd_array[i, 3:3+len(svd_dict[key])] = svd_dict[key]

    np.savez('stk_matrix_data', cond_array, svd_array,
             cond_array=cond_array, svd_array=svd_array)

    fig, ax = plt.subplots(1, 2)
    for key, value in svd_dict.items():
        i = key[1] * 2 + key[2]
        if key[0] == 8:
            ax[0].plot([i] * len(value), value, 'o')
        elif key[0] == 16:
            ax[1].plot([i] * len(value), value, 'o')
    plt.show()
    print('Done!')


if __name__ == '__main__':
    main()
