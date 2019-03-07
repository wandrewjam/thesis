from rolling_algorithms import load_deterministic_data, load_stochastic_data
import numpy as np


def process_data(det_file, sto_file):
    (bond_counts, v, om, avg_lengths, bond_mesh, force,
     torque, t_mesh) = load_deterministic_data(det_file)

    (count_array, v_array, om_array, force_array, torque_array, len_array,
     master_list, t_sample) = load_stochastic_data(sto_file)

    count_mean, v_mean, om_mean, force_mean, torque_mean = tuple(
        np.mean(a, axis=0) for a in (count_array, v_array, om_array,
                                     force_array, torque_array))

    count_std, v_std, om_std, force_std, torque_std = tuple(
        np.std(a, axis=0)for a in (count_array, v_array, om_array,
                                   force_array, torque_array))

    len_mean = np.nanmean(len_array, axis=0)
    len_std = np.nanstd(len_array, axis=0)

    if count_array.shape[0] < 20:
        (count_sampled, v_sampled, om_sampled, fc_sampled, tq_sampled,
         len_sampled) = (count_array, v_array, om_array, force_array,
                         torque_array, len_array)
    else:
        runs = np.random.choice(count_array.shape[0], 20, replace=False)
        (count_sampled, v_sampled, om_sampled, fc_sampled, tq_sampled,
         len_sampled) = tuple(el[runs, :] for el in (
            count_array, v_array, om_array, force_array, torque_array,
            len_array))

    return (bond_counts, v, om, avg_lengths, bond_mesh, force, torque, t_mesh,
            count_mean, count_std, v_mean, v_std, om_mean, om_std, force_mean,
            force_std, torque_mean, torque_std, len_mean, len_std, master_list,
            t_sample, count_sampled, v_sampled, om_sampled, fc_sampled,
            tq_sampled, len_sampled)


if __name__ == '__main__':
    import sys
    det_loadname, sto_loadname = sys.argv[1], sys.argv[2]

    (bond_counts, v, om, avg_lengths, bond_mesh, force, torque, t_mesh,
     count_mean, count_std, v_mean, v_std, om_mean, om_std, force_mean,
     force_std, torque_mean, torque_std, len_mean, len_std, master_list,
     t_sample, count_sampled, v_sampled, om_sampled, fc_sampled, tq_sampled,
     len_sampled) = process_data(det_loadname, sto_loadname)

    det_list = det_loadname.split('/')
    sto_list = sto_loadname.split('/')
    det_list[-1] = 'processed_' + det_list[-1]
    sto_list[-1] = 'processed_' + sto_list[-1]
    det_savename = '/'.join(det_list)
    sto_savename = '/'.join(sto_list)

    np.savez_compressed(det_savename, bond_counts, v, om, avg_lengths,
                        bond_mesh, force, torque, t_mesh,
                        bond_counts=bond_counts, v=v, om=om,
                        avg_lengths=avg_lengths, bond_mesh=bond_mesh,
                        force=force, torque=torque, t_mesh=t_mesh)

    np.savez_compressed(sto_savename, count_mean, count_std, v_mean, v_std,
                        om_mean, om_std, force_mean, force_std, torque_mean,
                        torque_std, len_mean, len_std, master_list, t_sample,
                        count_sampled, v_sampled, om_sampled, fc_sampled,
                        tq_sampled, len_sampled, count_mean=count_mean,
                        count_std=count_std, v_mean=v_mean, v_std=v_std,
                        om_mean=om_mean, om_std=om_std, force_mean=force_mean,
                        force_std=force_std, torque_mean=torque_mean,
                        torque_std=torque_std, len_mean=len_mean,
                        len_std=len_std, master_list=master_list,
                        t_sample=t_sample, count_sampled=count_sampled,
                        v_sampled=v_sampled, om_sampled=om_sampled,
                        fc_sampled=fc_sampled, tq_sampled=tq_sampled,
                        len_sampled=len_sampled)

    print('Wrote output to {:s} and {:s}'.format(det_savename, sto_savename))
