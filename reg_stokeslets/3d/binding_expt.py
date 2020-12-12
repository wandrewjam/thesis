import numpy as np
from motion_integration import (integrate_motion, nondimensionalize,
                                eps_picker, get_bond_lengths)
from resistance_matrix_test import generate_resistance_matrices
import matplotlib.pyplot as plt
import os
from scipy.io import savemat


def save_info(fname, seed, t_span, num_steps, n_nodes, a, b, adaptive, shear,
              l_sep, dimk0_on, dimk0_off, sig, sig_ts):
    with open(fname, 'w') as f:
        expt_info = ['seed, {}\n'.format(seed),
                     't_span, [{} {}]\n'.format(*t_span),
                     'num_steps, {}\n'.format(num_steps),
                     'n_nodes, {}\n'.format(n_nodes),
                     'a, {}\n'.format(a),
                     'b, {}\n'.format(b),
                     'adaptive, {}\n'.format(adaptive),
                     'shear, {}\n'.format(shear),
                     'l_sep, {}\n'.format(l_sep),
                     'dimk0_on, {}\n'.format(dimk0_on),
                     'dimk0_off, {}\n'.format(dimk0_off),
                     'sig, {}'.format(sig),
                     'sig_ts, {}'.format(sig_ts)
                     ]

        f.writelines(expt_info)


def main(expt_num=None, save_data=True, plot_data=False):
    save_dir = 'data/'
    # Read in previous file names
    file_i = [int(f[7:10]) for f in os.listdir(save_dir) if f[:7] == 'bd_expt']
    if expt_num is None:
        try:
            max_i = max(file_i)
        except ValueError:
            max_i = -1
    else:
        if expt_num in file_i:
            while True:
                cont = raw_input('Do you want to overwrite file {}?: (Y or N) '
                                 .format(save_dir + 'bd_expt{:03d}.npz'
                                         .format(expt_num)))
                if cont == 'Y':
                    break
                elif cont == 'N':
                    print('Exiting program')
                    return None
                print('Type Y or N')

    t_span = [0., 50.]
    num_steps = 250

    if expt_num % 3 == 0:
        one_side = False
        check_bonds = False
    elif expt_num % 3 == 1:
        one_side = True
        check_bonds = False
    elif expt_num % 3 == 2:
        one_side = True
        check_bonds = True

    if expt_num // 3 == 0:
        seed = 21554160
    elif expt_num // 3 == 1:
        seed = 215541690
    elif expt_num // 3 == 2:
        seed = 4072544895
    else:
        seed = np.random.randint(int('1'*32, 2)+1)
    np.random.seed(seed)

    def exact_vels(em):
        return np.zeros(6)

    expt = 3
    n_nodes = 8
    a, b = 1.5, .5
    domain = 'wall'
    adaptive = False
    if expt == 1:
        bonds = np.array([[0, 0., 0.]])
        init = np.array([.6, 0, 0, 1., 0, 0])
        receptors = np.array([[-.5, 0., 0.]])
    elif expt == 2:
        bonds = np.array([[0, 0., 0.]])
        init = np.array([1.55, 0, 0, 0, 0, -1.])
        receptors = np.array([[0., 1.5, 0.]])
    elif expt == 3:
        init = np.array([1.2, 0, 0, 1., 0, 0])
        receptors = np.load('Xb_0.26.npy')
        bonds = np.zeros(shape=(0, 3), dtype='float')
        # bonds = np.array([[np.argmax(receptors[:, 1]), 0., 0.]])
    else:
        raise ValueError('expt is not valid')

    # Define parameters
    shear = 100.
    l_sep = .1
    dimk0_on = 10.
    dimk0_off = 5.
    sig = 1e2
    sig_ts = 9.99e1
    t_sc, f_sc, lam, k0_on, k0_off, eta, eta_ts, kappa = nondimensionalize(
        l_scale=1, shear=shear, mu=4e-3, l_sep=l_sep, dimk0_on=dimk0_on,
        dimk0_off=dimk0_off, sig=sig, sig_ts=sig_ts, temp=310.)

    result = integrate_motion(t_span, num_steps, init, exact_vels, n_nodes, a, b, domain, adaptive=adaptive,
                              receptors=receptors, bonds=bonds, eta=eta, eta_ts=eta_ts, kappa=kappa, lam=lam,
                              k0_on=k0_on, k0_off=k0_off, check_bonds=check_bonds, one_side=one_side)

    t = result[9] * t_sc
    # t = np.linspace(t_span[0], t_span[1], num=num_steps+1) * t_sc
    mask = result[0] > 0

    center = np.stack(result[:3])

    if plot_data:
        fig, axs = plt.subplots(3, sharex='all', figsize=(6, 8))
        x_ln = axs[0].plot(t[mask], result[0][mask], label='x')
        ax_tw = axs[0].twinx()
        z_ln = ax_tw.plot(t[mask], result[2][mask], color='tab:orange', label='z')
        axs[0].set_ylabel('Center of mass ($\\mu$m)')
        lns = x_ln + z_ln
        labs = [l.get_label() for l in lns]
        axs[0].legend(lns, labs)

        axs[1].plot(t[mask], result[3][2, 0, mask])
        axs[1].set_ylabel('Orientation component')
        axs[1].legend(['e_mz'])

        # fig, ax = plt.subplots()
        # if expt == 3:
        # Figure out how to plot bonds individually (and only bonds that exist at a particular time point)
        # Use a dict to track individual bonds over time
        bond_dict = {}
        for i, t_i in enumerate(t):
            rmat = result[3][:, :, i]
            true_receptors = center[:, [i]].T + np.dot(receptors, rmat.T)
            bond_lens = get_bond_lengths(result[8][i], true_receptors)
            for j in range(len(result[8][i])):
                key = tuple(result[8][i][j])
                try:
                    bond_dict[key] = (np.append(bond_dict[key][0], t_i),
                                      np.append(bond_dict[key][1],
                                                bond_lens[j]))
                except KeyError:
                    bond_dict[key] = (t_i, bond_lens[j])
        for times, bond_len in bond_dict.values():
            axs[2].plot(times, bond_len)
        # else:
        #     lengths = np.linalg.norm(
        #         center[:, None, :] + np.dot(result[3].transpose((2, 0, 1)),
        #                                     receptors.T).transpose((1, 2, 0)),
        #         axis=0)
        #     axs[2].plot(t[mask], lengths.T[mask])
        # axs[2].set_xlim(t[0], t[-1])
        axs[2].set_title('Bond length')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Length ($\\mu$m)')
        plt.tight_layout()
        plt.show()

    if save_data:
        file_fmt = 'bd_expt{:03d}'.format(expt_num)

        x, y, z, r_matrices = result[:4]
        bond_history = result[8]

        max_bonds = len(max(*bond_history, key=len))
        padded_bonds = [np.pad(bd, pad_width=((0, max_bonds - bd.shape[0]),
                                              (0, 0)),
                               mode='constant', constant_values=-1)
                        for bd in bond_history]
        bond_array = np.stack(padded_bonds, axis=-1)

        np.savez(save_dir + file_fmt, t, x, y, z, r_matrices, bond_array,
                 receptors, t=t, x=x, y=y, z=z, r_matrices=r_matrices,
                 bond_array=bond_array, receptors=receptors)
        savemat(save_dir + file_fmt,
                {'t': t, 'x': x, 'y': y, 'z': z, 'R': r_matrices,
                 'bond_array': bond_array, 'receptors': receptors})
        fname = save_dir + 'info_' + file_fmt + '.txt'
        save_info(fname, seed, t_span, num_steps, n_nodes, a, b, adaptive,
                  shear, l_sep, dimk0_on, dimk0_off, sig, sig_ts)

    bond_num = [bond.shape[0] for bond in result[8]]
    print(bond_num)
    theta = np.arctan2(result[3][2, 0, mask][-1], result[3][1, 0, mask][-1])
    phi = np.arccos(result[3][0, 0, mask][-1])
    eps = eps_picker(n_nodes, a=a, b=b)
    shear_f, shear_t = generate_resistance_matrices(
        eps, n_nodes, a=a, b=b, domain=domain, distance=center[0, mask][-1],
        theta=theta, phi=phi)[-2:]
    print(np.stack([shear_f, shear_t]))
    print('Seed = ' + str(seed))
    print('Done!')


if __name__ == '__main__':
    import sys

    try:
        expt_num = int(sys.argv[1])
    except IndexError:
        expt_num = None

    main(expt_num)
