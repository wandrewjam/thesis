import numpy as np
import matplotlib.pyplot as plt
from os import path
import pickle as pkl
from binding_expt import parse_file, save_info, save_rng
from motion_integration import integrate_motion, nondimensionalize, get_bond_lengths, find_min_separation, repulsive_force
from scipy.io import savemat


def continue_integration(filename, t_end=None, i_start=-1, debug=False, save_data=True):
    data_dir = path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    pars = parse_file(filename)

    try:
        with np.load(data_dir + filename + '.npz') as data:
            x = data['x']
            y = data['y']
            z = data['z']
            t = data['t']
            if t[-1] > 3.0001:
                t *= .01
            r_matrices = data['r_matrices']
            if r_matrices.shape[-1] == 3:
                r_matrices = r_matrices.transpose(1, 2, 0)
            receptors = data['receptors']
            bond_array = data['bond_array']
            rng_draws = data['draws']
    except FileNotFoundError:
        x = np.array([pars['x1']])
        y = np.array([0.])
        z = np.array([0.])
        t = np.zeros(1)

        theta = np.arctan2(pars['emz'], pars['emy'])
        phi = np.arccos(pars['emx'])
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        r_matrix = np.array([[cp, -sp, 0],
                             [ct * sp, ct * cp, -st],
                             [st * sp, st * cp, ct]])
        r_matrices = np.expand_dims(r_matrix, axis=-1)
        receptors = np.load(path.expanduser(
            '~/thesis/reg_stokeslets/src/3d/Xb_0.26.npy'))
        try:
            receptor_multiplier = pars['receptor_multiplier']
        except KeyError:
            receptor_multiplier = 1

        receptors = np.repeat(receptors, repeats=receptor_multiplier, axis=0)
        bond_array = np.zeros(shape=(0, 3, 1), dtype='float')
        rng_draws = np.zeros(shape=(1,), dtype=int)

    # if t_end < t[-1]:
    #     while True:
    #         response = input('Data will be lost. Continue? ')
    #         if response.lower() == 'y':
    #             break
    #         elif response.lower() == 'n':
    #             print('Exiting')
    #             return -1
    #         else:
    #             print('Please enter y or n')

    if t_end is None:
        t_end = pars['t_end']
        
    if np.abs(t_end - t[i_start]) < 1e-12:
        print('Simulation {} already completed'.format(filename))
        return

    assert t_end > t[i_start]

    if debug:
        import pdb
        pdb.set_trace()

    rng = np.random.RandomState(pars['seed'])
    rng.random(size=np.cumsum(rng_draws)[i_start])

    old_x, old_y, old_z = x[:i_start], y[:i_start], z[:i_start]
    old_t = t[:i_start]
    old_rmat = r_matrices[..., :i_start]
    old_bond_array = bond_array[..., :i_start]
    old_draws = rng_draws[:i_start]

    # Define arguments to the integrate_motion function
    dt = (pars['t_end'] - pars['t_start']) / pars['num_steps']
    num_steps = np.ceil((t_end - t[i_start]) / dt - 1e-8).astype('int')
    # num_steps = 10

    center = np.array([x[i_start], y[i_start], z[i_start]])
    init = np.concatenate((center, r_matrices[..., i_start].reshape((9,))))
    n_nodes = pars['n_nodes']
    adaptive = pars['adaptive']
    # order = '2nd'
    # proc = 1
    bonds = bond_array[bond_array[:, 0, i_start] >= 0, :, i_start]

    domain = 'wall'
    a, b = pars['a'], pars['b']
    shear = pars['shear']

    sig1, sig_ts1, l_sep1 = pars['sig'], pars['sig_ts'], pars['l_sep']
    dimk0_on, dimk0_off = pars['dimk0_on'], pars['dimk0_off']
    dimk0_on2 = pars['dimk0_on2']
    check_bonds, one_side = pars['check_bonds'], pars['one_side']

    kratio = 0.4
    nd_parameters = nondimensionalize(
        l_scale=1, shear=shear, mu=4e-3, l_sep=l_sep1, dimk0_on=dimk0_on,
        dimk0_off=dimk0_off, dimk0_on2=dimk0_on2, sig=sig1, sig_ts=sig_ts1,
        temp=310.)

    (t_sc, f_sc, lam1, k0_on, k0_off, k0_on2, eta, eta_ts1, kappa1,
     gammp, kn0off, ki0off, yn, yi) = nd_parameters

    nd_start, nd_end = t[i_start] / t_sc, t_end / t_sc

    def exact_vels(em):
        return np.zeros(6)

    print('Running simulation {}'.format(filename))

    result = integrate_motion([nd_start, nd_end], num_steps, init, exact_vels,
                              n_nodes, a, b, domain, adaptive=adaptive,
                              receptors=receptors, bonds=bonds, eta=eta,
                              eta_ts1=eta_ts1, kappa1=kappa1, lam1=lam1,
                              eta_ts2=eta_ts1, kappa2=kappa1, lam2=lam1,
                              k0_on=k0_on, k0_off=k0_off, k0_on2=k0_on2,
                              check_bonds=check_bonds, one_side=one_side,
                              save_file=filename + '_cont', rng=rng, t_sc=t_sc,
                              kratio=kratio, gammp=gammp, yn=yn, yi=yi,
                              knoff=kn0off, kioff=ki0off)

    t = result[9] * t_sc
    center = np.stack(result[:3])

    x, y, z, r_matrices = result[:4]
    bond_history = result[8]
    draws = result[10]

    max_bonds = len(max(*bond_history, key=len))
    max_bonds = max([max_bonds, old_bond_array.shape[0]])
    old_padded = np.pad(old_bond_array[..., :],
                        pad_width=((0, max_bonds - old_bond_array.shape[0]),
                                   (0, 0), (0, 0)),
                        mode='constant', constant_values=-1)

    padded_bonds = [np.pad(bd, pad_width=((0, max_bonds - bd.shape[0]),
                                          (0, 0)),
                           mode='constant', constant_values=-1)
                    for bd in bond_history]
    bond_array = np.stack(padded_bonds, axis=-1)

    new_t = np.concatenate((old_t, t))
    new_x = np.concatenate((old_x, x))
    new_y = np.concatenate((old_y, y))
    new_z = np.concatenate((old_z, z))
    new_rmat = np.concatenate((old_rmat, r_matrices), axis=-1)
    new_bond_array = np.concatenate((old_padded, bond_array), axis=-1)
    new_draws = np.concatenate((old_draws, draws[1:]))

    if save_data:
        new_filename = filename + '_cont'
        np.savez(data_dir + new_filename, t=new_t, x=new_x, y=new_y, z=new_z,
                 r_matrices=new_rmat, bond_array=new_bond_array,
                 receptors=receptors, draws=new_draws)
        savemat(data_dir + new_filename + '.mat',
                {'t': new_t, 'x': new_x, 'y': new_y, 'z': new_z, 'R': new_rmat,
                 'bond_array': new_bond_array, 'receptors': receptors,
                 'draws': new_draws})
        save_info(new_filename, seed=pars['seed'], t_start=pars['t_start'],
                  t_end=pars['t_end'], num_steps=num_steps, n_nodes=n_nodes,
                  a=a, b=b, adaptive=adaptive, shear=shear, l_sep=l_sep1,
                  dimk0_on=dimk0_on, dimk0_off=dimk0_off, dimk0_on2=dimk0_on2,
                  sig=sig1, sig_ts=sig_ts1, one_side=one_side,
                  check_bonds=check_bonds)

    # t = new_t
    # x = new_x
    # y = new_y
    # z = new_z
    # r_matrices = new_rmat
    # bond_array = new_bond_array
    #
    # mask = x > 0
    #
    # fig, axs = plt.subplots(5, sharex='all', figsize=(6.5, 9))
    # x_ln = axs[0].plot(t[mask], x[mask], label='x')
    # ax_tw = axs[0].twinx()
    # z_ln = ax_tw.plot(t[mask], z[mask], color='tab:orange', label='z')
    # axs[0].set_ylabel('Center of mass ($\\mu$m)')
    # lns = x_ln + z_ln
    # labs = [l.get_label() for l in lns]
    # axs[0].legend(lns, labs)
    # axs[1].plot(t[mask], r_matrices[2, 0, mask])
    # axs[1].set_ylabel('Orientation')
    # axs[1].legend(['e_mz'])
    # center = np.stack([x, y, z])
    # # fig, ax = plt.subplots()
    # # if expt == 3:
    # # Figure out how to plot bonds individually (and only bonds that exist at a particular time point)
    # # Use a dict to track individual bonds over time
    # bond_dict = {}
    # for i, t_i in enumerate(t):
    #     rmat = r_matrices[:, :, i]
    #     true_receptors = center[:, [i]].T + np.dot(receptors, rmat.T)
    #     current_mask = bond_array[:, 0, i] != -1
    #     current_bonds = bond_array[current_mask, :, i]
    #     bond_lens = get_bond_lengths(current_bonds, true_receptors)
    #     z_rec = true_receptors[current_bonds[:, 0].astype('int'), 2] - z[i]
    #     z_cmp = current_bonds[:, 2] - true_receptors[
    #         current_bonds[:, 0].astype('int'), 2]
    #     x_cmp = true_receptors[current_bonds[:, 0].astype('int'), 0]
    #     y_cmp = current_bonds[:, 1] - true_receptors[
    #         current_bonds[:, 0].astype('int'), 1]
    #
    #     for j in range(len(current_bonds)):
    #         key = tuple(current_bonds[j, :])
    #         try:
    #             bond_dict[key] = (
    #                 np.append(bond_dict[key][0], t_i),
    #                 np.append(bond_dict[key][1], bond_lens[j]),
    #                 np.append(bond_dict[key][2], z_rec[j]),
    #                 np.append(bond_dict[key][3], z_cmp[j]),
    #                 np.append(bond_dict[key][4], x_cmp[j]),
    #                 np.append(bond_dict[key][5], y_cmp[j]))
    #         except KeyError:
    #             bond_dict[key] = (t_i, bond_lens[j], z_rec[j], z_cmp[j],
    #                               x_cmp[j], y_cmp[j])
    # for times, bond_len, z_rec, z_cmp, x_cmp, y_cmp in bond_dict.values():
    #     line, = axs[2].plot(times, bond_len)
    #     axs[3].plot(times, z_rec, color=line.get_color())
    #     axs[4].plot(times, z_cmp, color=line.get_color())
    # # else:
    # #     lengths = np.linalg.norm(
    # #         center[:, None, :] + np.dot(result[3].transpose((2, 0, 1)),
    # #                                     receptors.T).transpose((1, 2, 0)),
    # #         axis=0)
    # #     axs[2].plot(t[mask], lengths.T[mask])
    # # axs[2].set_xlim(t[0], t[-1])
    # # axs[2].set_title('Bond length')
    # # axs[2].set_xlabel('Time (s)')
    # # Set horizontal reference line
    # t_ref = [t[0], t[-1]]
    # axs[2].plot(t_ref, [0.1, 0.1], linewidth=.7, color='k')
    # axs[3].plot(t_ref, [0, 0], linewidth=.7, color='k')
    # axs[4].plot(t_ref, [0, 0], linewidth=.7, color='k')
    #
    # axs[2].set_ylabel('Length ($\\mu$m)')
    # axs[3].set_ylabel('$z$ difference ($\\mu$m)')
    # axs[4].set_ylabel('$z$ difference ($\\mu$m)')
    # axs[-1].set_xlabel('Time (s)')
    #
    # # plt.savefig(plot_dir + filename + '_1', bbox_inches='tight')
    # plt.tight_layout()
    # plt.show()
    #
    # # Compute separation distance and rep force for each time step
    # sep_distance = np.zeros(shape=x.shape)
    # rep_force = np.zeros(shape=x.shape)
    # for i in range(len(x)):
    #     ix, iy, iz = x[i], y[i], z[i]
    #     irmat = r_matrices[:, :, i]
    #     sep_distance[i] = find_min_separation(ix, irmat[:, 0])
    #     rep_force[i] = repulsive_force(ix, irmat[:, 0])[1]
    # fig, axs = plt.subplots(nrows=3)
    # axs[0].semilogy(t, sep_distance)
    # axs[1].plot(t, rep_force)
    # for times, bond_len, z_rec, z_cmp, x_cmp, y_cmp in bond_dict.values():
    #     axs[2].plot(times, x_cmp)
    #     axs[2].plot(times, y_cmp, linestyle='--')
    #
    # # plt.savefig(plot_dir + filename + '_2', bbox_inches='tight')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    import sys

    filename = sys.argv[1]
    continue_integration(filename)
