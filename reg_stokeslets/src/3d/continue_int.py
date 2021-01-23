import numpy as np
import matplotlib.pyplot as plt
from os import path
import pickle as pkl
from binding_expt import parse_file, save_info, save_rng
from motion_integration import integrate_motion, nondimensionalize, get_bond_lengths, find_min_separation, repulsive_force
from scipy.io import savemat


def continue_integration(filename, t_end, i_start=-1):
    data_dir = path.expanduser('~/thesis/reg_stokeslets/data/bd_run/')
    pars = parse_file(filename)

    try:
        with open(data_dir + filename + '.pkl', 'rb') as f:
            rng_history = pkl.load(f)
    except UnicodeDecodeError as e:
        with open(data_dir + filename + '.pkl', 'rb') as f:
            rng_history = pkl.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', filename, ':', e)
        raise

    with np.load(data_dir + filename + '.npz') as data:
        x = data['x']
        y = data['y']
        z = data['z']
        t = data['t']
        r_matrices = data['r_matrices']
        receptors = data['receptors']
        bond_array = data['bond_array']

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

    assert t_end > t[i_start]

    np.random.set_state(rng_history[i_start])

    old_x, old_y, old_z = x[:i_start], y[:i_start], z[:i_start]
    old_t = t[:i_start]
    old_rmat = r_matrices[..., :i_start]
    old_bond_array = bond_array[..., :i_start]
    old_rng = rng_history[:i_start]

    # Define arguments to the integrate_motion function
    dt = (pars['t_end'] - pars['t_start']) / pars['num_steps']
    num_steps = np.ceil((t_end - t[i_start]) / dt - 1e-8).astype('int')

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

    sig, sig_ts, l_sep = pars['sig'], pars['sig_ts'], pars['l_sep']
    dimk0_on, dimk0_off = pars['dimk0_on'], pars['dimk0_off']
    check_bonds, one_side = pars['check_bonds'], pars['one_side']

    t_sc, f_sc, lam, k0_on, k0_off, eta, eta_ts, kappa = nondimensionalize(
        l_scale=1, shear=shear, mu=4e-3, l_sep=l_sep, dimk0_on=dimk0_on,
        dimk0_off=dimk0_off, sig=sig, sig_ts=sig_ts, temp=310.)

    nd_start, nd_end = t[i_start] / t_sc, t_end / t_sc

    def exact_vels(em):
        return np.zeros(6)

    result = integrate_motion(
        [nd_start, nd_end], num_steps, init, exact_vels, n_nodes, a, b, domain,
        adaptive=adaptive, receptors=receptors, bonds=bonds, eta=eta,
        eta_ts=eta_ts, kappa=kappa, lam=lam, k0_on=k0_on, k0_off=k0_off,
        check_bonds=check_bonds, one_side=one_side)

    t = result[9] * t_sc
    center = np.stack(result[:3])

    x, y, z, r_matrices = result[:4]
    bond_history = result[8]
    rng_states = result[10]

    max_bonds = len(max(*bond_history, key=len))
    max_bonds = max([max_bonds, old_bond_array.shape[0]])
    old_padded = np.pad(old_bond_array,
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
    new_bond_array =  np.concatenate((old_padded, bond_array), axis=-1)
    new_rng = old_rng + rng_states

    # np.savez(data_dir + filename, new_t, new_x, new_y, new_z, new_rmat,
    #          new_bond_array, receptors, t=new_t, x=new_x, y=new_y, z=new_z,
    #          r_matrices=new_rmat, bond_array=new_bond_array,
    #          receptors=receptors)
    # savemat(data_dir + filename,
    #         {'t': new_t, 'x': new_x, 'y': new_y, 'z': new_z, 'R': new_rmat,
    #          'bond_array': new_bond_array, 'receptors': receptors})
    # save_info(filename, pars['seed'], pars['t_start'], pars['t_end'], num_steps, n_nodes, a, b,
    #           adaptive, shear, l_sep, dimk0_on, dimk0_off,
    #           sig, sig_ts, one_side, check_bonds)
    # save_rng(filename, rng_states)

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
    continue_integration('test_expt', 0.2)