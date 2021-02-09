import numpy as np
from motion_integration import (integrate_motion, nondimensionalize,
                                eps_picker, get_bond_lengths, evaluate_motion_equations)
from resistance_matrix_test import generate_resistance_matrices
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
from timeit import default_timer as timer


def parse_file(filename):
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
    parlist = [('filename', filename)]
    with open(txt_dir + filename + '.txt', 'r') as f:
        while True:
            command = f.readline().split()
            if len(command) < 1:
                continue
            if command[0] == 'done':
                break

            key, value = command
            if key in ['seed', 'num_steps', 'n_nodes']:
                parlist.append((key, int(value)))
            elif key in ['adaptive', 'one_side', 'check_bonds']:
                parlist.append((key, 'True' == value))
            elif key == 'order':
                parlist.append((key, value))
            else:
                parlist.append((key, float(value)))
    return dict(parlist)


def save_info(filename, seed, t_start, t_end, num_steps, n_nodes, a, b,
              adaptive, shear, l_sep, dimk0_on, dimk0_off, sig, sig_ts,
              one_side, check_bonds, x1=1.2, x2=0., x3=0., emx=1., emy=.0, emz=0., order='2nd'):
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
    with open(txt_dir + filename + '.txt', 'w') as f:
        expt_info = ['seed {}\n'.format(seed),
                     't_start {}\n'.format(t_start),
                     't_end {}\n'.format(t_end),
                     'num_steps {}\n'.format(num_steps),
                     'n_nodes {}\n'.format(n_nodes),
                     'order {}\n'.format(order),
                     'adaptive {}\n'.format(adaptive),
                     'one_side {}\n'.format(one_side),
                     'check_bonds {}\n'.format(check_bonds),
                     'a {}\n'.format(a),
                     'b {}\n'.format(b),
                     'x1 {}\n'.format(x1),
                     'x2 {}\n'.format(x2),
                     'x3 {}\n'.format(x3),
                     'emx {}\n'.format(emx),
                     'emy {}\n'.format(emy),
                     'emz {}\n'.format(emz),
                     'shear {}\n'.format(shear),
                     'l_sep {}\n'.format(l_sep),
                     'dimk0_on {}\n'.format(dimk0_on),
                     'dimk0_off {}\n'.format(dimk0_off),
                     'sig {}\n'.format(sig),
                     'sig_ts {}\n'.format(sig_ts),
                     '\n', 'done\n'
                     ]

        f.writelines(expt_info)


def save_rng(filename, rng_states):
    import pickle
    save_dir = os.path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    with open(save_dir + filename + '.pkl', 'wb') as f:
        pickle.dump(rng_states, f)


def main(filename, expt_num=None, save_data=True, plot_data=False, t_start=0.,
         t_end=50., num_steps=250, seed=None, n_nodes=8, adaptive=False, a=1.5,
         b=.5, shear=100., l_sep=0.1, dimk0_on=10., dimk0_off=5., sig=1e4,
         sig_ts=9.99e3, one_side=False, check_bonds=False, x1=1.2, x2=0.,
         x3=0., emx=1., emy=0., emz=0, order='2nd'):
    save_dir = os.path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
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

    # t_span = [0., 50.]
    # num_steps = 250

    # if expt_num % 3 == 0:
    #     one_side = False
    #     check_bonds = False
    # elif expt_num % 3 == 1:
    #     one_side = True
    #     check_bonds = False
    # elif expt_num % 3 == 2:
    #     one_side = True
    #     check_bonds = True

    # if expt_num // 3 == 0:
    #     seed = 21554160
    # elif expt_num // 3 == 1:
    #     seed = 215541690
    # elif expt_num // 3 == 2:
    #     seed = 4072544895
    # elif expt_num // 3 == 27:
    #     seed = 803402144
    # else:
    #     seed = np.random.randint(int('1'*32, 2)+1)
    if seed is None:
        seed = np.random.randint(int('1'*32, 2)+1)
    np.random.seed(seed)

    def exact_vels(em):
        return np.zeros(6)

    expt = 3
    # n_nodes = 8
    # a, b = 1.5, .5
    domain = 'wall'
    # adaptive = False
    if expt == 1:
        bonds = np.array([[0, 0., 0.]])
        init = np.array([.6, 0, 0, 1., 0, 0])
        receptors = np.array([[-.5, 0., 0.]])
    elif expt == 2:
        bonds = np.array([[0, 0., 0.]])
        init = np.array([1.55, 0, 0, 0, 0, -1.])
        receptors = np.array([[0., 1.5, 0.]])
    elif expt == 3:
        init = np.array([x1, x2, x3, emx, emy, emz])
        receptors = np.load(os.path.expanduser('~/thesis/reg_stokeslets/src/3d/Xb_0.26.npy'))
        bonds = np.zeros(shape=(0, 3), dtype='float')
        # bonds = np.array([[np.argmax(receptors[:, 1]), 0., 0.]])
    else:
        raise ValueError('expt is not valid')

    # Define parameters
    # shear = 100.
    # l_sep = .1
    # dimk0_on = 10.
    # dimk0_off = 5.
    # sig = 1e4
    # sig_ts = 9.99e3
    t_sc, f_sc, lam, k0_on, k0_off, eta, eta_ts, kappa = nondimensionalize(
        l_scale=1, shear=shear, mu=4e-3, l_sep=l_sep, dimk0_on=dimk0_on,
        dimk0_off=dimk0_off, sig=sig, sig_ts=sig_ts, temp=310.)

    nd_start, nd_end = t_start / t_sc, t_end / t_sc
    start = timer()
    result = integrate_motion(
        [nd_start, nd_end], num_steps, init, exact_vels, n_nodes, a, b, domain, order=order,
        adaptive=adaptive, receptors=receptors, bonds=bonds, eta=eta,
        eta_ts=eta_ts, kappa=kappa, lam=lam, k0_on=k0_on, k0_off=k0_off,
        check_bonds=check_bonds, one_side=one_side)
    end = timer()

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
        # file_fmt = 'bd_expt{:03d}'.format(expt_num)

        x, y, z, r_matrices = result[:4]
        bond_history = result[8]
        rng_states = result[10]

        max_bonds = len(max(*bond_history, key=len))
        padded_bonds = [np.pad(bd, pad_width=((0, max_bonds - bd.shape[0]),
                                              (0, 0)),
                               mode='constant', constant_values=-1)
                        for bd in bond_history]
        bond_array = np.stack(padded_bonds, axis=-1)

        np.savez(save_dir + filename, t, x, y, z, r_matrices, bond_array,
                 receptors, t=t, x=x, y=y, z=z, r_matrices=r_matrices,
                 bond_array=bond_array, receptors=receptors)
        savemat(save_dir + filename + '.mat',
                {'t': t, 'x': x, 'y': y, 'z': z, 'R': r_matrices,
                 'bond_array': bond_array, 'receptors': receptors})
        save_info(filename, seed, t_start, t_end, num_steps, n_nodes, a, b,
                  adaptive, shear, l_sep, dimk0_on, dimk0_off,
                  sig, sig_ts, one_side, check_bonds)
        save_rng(filename, rng_states)

    bond_num = [bond.shape[0] for bond in result[8]]
    # print(bond_num)
    theta = np.arctan2(result[3][2, 0, mask][-1], result[3][1, 0, mask][-1])
    phi = np.arccos(result[3][0, 0, mask][-1])
    eps = eps_picker(n_nodes, a=a, b=b)
    shear_f, shear_t = generate_resistance_matrices(
        eps, n_nodes, a=a, b=b, domain=domain, distance=center[0, mask][-1],
        theta=theta, phi=phi)[-2:]
    print(filename + ':')
    print('Integration took {} seconds'.format(end - start))
    print('RHS Evaluations: {}'.format(evaluate_motion_equations.counter))
    print()
    # print(np.stack([shear_f, shear_t]))
    # print('Seed = ' + str(seed))
    # print('Done!')


if __name__ == '__main__':
    import sys

    pars = parse_file(sys.argv[1])
    main(**pars)
