from rolling_algorithms import load_deterministic_data, load_stochastic_data
import numpy as np


def extract_stats(stat, *args):
    """ Extract and return stat from a set of experiments """
    return tuple(stat(a, axis=0) for a in args)


def quart1(arr, axis):
    """ Find the 1st quartile """
    return np.quantile(a=arr, q=.25, axis=axis)


def quart3(arr, axis):
    """ Find the 3rd quartile """
    return np.quantile(a=arr, q=.75, axis=axis)


def process_data(det_file, sto_file):
    """" Extract statistics from a set of experiments """
    (bond_counts, v, om, avg_lengths, bond_mesh, force,
     torque, t_mesh) = load_deterministic_data(det_file)

    (count_array, v_array, om_array, force_array, torque_array, len_array,
     master_list, t_sample) = load_stochastic_data(sto_file)

    count_mean, v_mean, om_mean, force_mean, torque_mean = extract_stats(
        np.mean, count_array, v_array, om_array, force_array, torque_array)

    count_std, v_std, om_std, force_std, torque_std = extract_stats(
        np.std, count_array, v_array, om_array, force_array, torque_array)

    count_med, v_med, om_med, force_med, torque_med = extract_stats(
        np.median, count_array, v_array, om_array, force_array, torque_array)

    count_1q, v_1q, om_1q, force_1q, torque_1q = extract_stats(
        quart1, count_array, v_array, om_array, force_array, torque_array)

    count_3q, v_3q, om_3q, force_3q, torque_3q = extract_stats(
        quart3, count_array, v_array, om_array, force_array, torque_array)

    len_mean = np.nanmean(len_array, axis=0)
    len_std = np.nanstd(len_array, axis=0)
    len_med = np.nanmedian(len_array, axis=0)
    len_1q = np.nanquantile(len_array, q=.25, axis=0)
    len_3q = np.nanquantile(len_array, q=.75, axis=0)

    if count_array.shape[0] < 20:
        runs = np.arange(count_array.shape[0])
    else:
        runs = np.random.choice(count_array.shape[0], 20, replace=False)

    (count_sampled, v_sampled, om_sampled, fc_sampled, tq_sampled,
     len_sampled) = tuple(el[runs, :] for el in (
        count_array, v_array, om_array, force_array, torque_array,
        len_array))

    return (bond_counts, v, om, avg_lengths, bond_mesh, force, torque, t_mesh,
            count_mean, count_std, count_med, count_1q, count_3q, v_mean,
            v_std, v_med, v_1q, v_3q, om_mean, om_std, om_med, om_1q, om_3q,
            force_mean, force_std, force_med, force_1q, force_3q, torque_mean,
            torque_std, torque_med, torque_1q, torque_3q, len_mean, len_std,
            len_med, len_1q, len_3q, master_list, t_sample, count_sampled,
            v_sampled, om_sampled, fc_sampled, tq_sampled, len_sampled, runs)


if __name__ == '__main__':
    import sys
    det_loadname, sto_loadname = sys.argv[1], sys.argv[2]

    (bond_counts, v, om, avg_lengths, bond_mesh, force, torque, t_mesh,
     count_mean, count_std, count_med, count_1q, count_3q, v_mean,
     v_std, v_med, v_1q, v_3q, om_mean, om_std, om_med, om_1q, om_3q,
     force_mean, force_std, force_med, force_1q, force_3q, torque_mean,
     torque_std, torque_med, torque_1q, torque_3q, len_mean, len_std,
     len_med, len_1q, len_3q, master_list, t_sample, count_sampled,
     v_sampled, om_sampled, fc_sampled, tq_sampled, len_sampled,
     runs) = process_data(det_loadname, sto_loadname)

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

    np.savez_compressed(
        sto_savename, count_mean, count_std, count_med, count_1q, count_3q,
        v_mean, v_std, v_med, v_1q, v_3q, om_mean, om_std, om_med, om_1q,
        om_3q, force_mean, force_std, force_med, force_1q, force_3q,
        torque_mean, torque_std, torque_med, torque_1q, torque_3q, len_mean,
        len_std, len_med, len_1q, len_3q, master_list, t_sample, count_sampled,
        v_sampled, om_sampled, fc_sampled, tq_sampled, len_sampled, runs,
        count_mean=count_mean, count_std=count_std, count_med=count_med,
        count_1q=count_1q, count_3q=count_3q, v_mean=v_mean, v_std=v_std,
        v_med=v_med, v_1q=v_1q, v_3q=v_3q, om_mean=om_mean, om_std=om_std,
        om_med=om_med, om_1q=om_1q, om_3q=om_3q, force_mean=force_mean,
        force_std=force_std, force_med=force_med, force_1q=force_1q,
        force_3q=force_3q, torque_mean=torque_mean, torque_std=torque_std,
        torque_med=torque_med, torque_1q=torque_1q, torque_3q=torque_3q,
        len_mean=len_mean, len_std=len_std, len_med=len_med, len_1q=len_1q,
        len_3q=len_3q, master_list=master_list, t_sample=t_sample,
        count_sampled=count_sampled, v_sampled=v_sampled,
        om_sampled=om_sampled, fc_sampled=fc_sampled, tq_sampled=tq_sampled,
        len_sampled=len_sampled, runs=runs
    )

    print('Wrote output to {:s} and {:s}'.format(det_savename, sto_savename))
