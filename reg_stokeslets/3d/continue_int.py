import numpy as np
import pickle as pkl
from binding_expt import parse_file, save_info, save_rng
from motion_integration import integrate_motion, nondimensionalize
from scipy.io import savemat


def continue_integration(filename, t_end, i_start=-1):
    import os
    data_dir = os.path.expanduser('~/thesis/reg_stokeslets/3d/data/')
    pars = parse_file(filename)

    with open(data_dir + filename + '.pkl', 'rb') as f:
        rng_history = pkl.load(f)

    with np.load(data_dir + filename + '.npz') as data:
        x = data['x']
        y = data['y']
        z = data['z']
        t = data['t']
        r_matrices = data['r_matrices']
        receptors = data['receptors']
        bond_array = data['bond_array']

    if t_end < t[-1]:
        while True:
            response = raw_input('Data will be lost. Continue? ')
            if response.lower() == 'y':
                break
            elif response.lower() == 'n':
                print('Exiting')
                return -1
            else:
                print('Please enter y or n')

    assert t_end > t[i_start]

    np.random.set_state(rng_history[i_start])

    old_x, old_y, old_z = x[:i_start], y[:i_start], z[:i_start]
    old_t = t[:i_start]
    old_rmat = r_matrices[..., :i_start]
    old_bond_array = bond_array[..., :i_start]
    old_rng = rng_history[:i_start]

    # Define arguments to the integrate_motion function
    dt = (t_end - t[i_start]) / pars['num_steps']
    num_steps = np.ceil((t_end - t[i_start]) / dt).astype('int')

    center = np.array([x[i_start], y[i_start], z[i_start]])
    init = np.concatenate((center, r_matrices[:, 0, i_start]))
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

    np.savez(data_dir + filename, new_t, new_x, new_y, new_z, new_rmat,
             new_bond_array, receptors, t=new_t, x=new_x, y=new_y, z=new_z,
             r_matrices=new_rmat, bond_array=new_bond_array,
             receptors=receptors)
    savemat(data_dir + filename,
            {'t': new_t, 'x': new_x, 'y': new_y, 'z': new_z, 'R': new_rmat,
             'bond_array': new_bond_array, 'receptors': receptors})
    save_info(filename, pars['seed'], pars['t_start'], pars['t_end'], num_steps, n_nodes, a, b,
              adaptive, shear, l_sep, dimk0_on, dimk0_off,
              sig, sig_ts, one_side, check_bonds)
    save_rng(filename, rng_states)


if __name__ == '__main__':
    continue_integration('test_expt', 0.2)
