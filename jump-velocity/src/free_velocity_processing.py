import numpy as np
from scipy.stats import f_oneway, kruskal, expon, sem, gaussian_kde
from scipy.optimize import minimize
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
# from matplotlib import cm
from extract_trajectory_data import (extract_state_data, load_trajectories,
                                     process_trajectory, truncate_trajectory)
from simulate import experiment


def main():
    # collagen = ['hcp', 'ccp', 'hcw', 'ccw']
    # collagen = ['hcw', 'ccw']
    # fibrinogen = ['hfp', 'ffp', 'hfw', 'ffw']
    fibrinogen = ['hfw', 'ffw']
    # vWF = ['hvp', 'vvp']
    colors = {'hfw': 'C1', 'ffw': 'C2'}
    # colors = {'hcp': 'C1', 'ccp': 'C2', 'hcw': 'C3', 'ccw': 'C4'}
    # colors = {'hcw': 'g', 'ccw': 'r'}
    experiments = load_trajectories(fibrinogen)
    result = get_free_velocities(experiments, absolute_pause_threshold=1.)
    (avg_free_vels_dict, distances_dict, times_dict, steps_dict, dwells_dict,
     ndwells_dict, vels_dict, escape_dict) = result

    Vstar = 40.
    average_distance = 5.

    k_off_dict = dict([(expt, 1./np.mean(dwell)) for expt, dwell
                       in dwells_dict.items()])
    k_tot_dict = dict([(expt, 1./np.mean(step)) for expt, step
                       in steps_dict.items()])
    a_dict = dict([(expt, 1 - 1./np.mean(ndwell)) for expt, ndwell
                   in ndwells_dict.items()])
    k_on_dict = dict([(expt, a_dict[expt] * k_tot) for expt, k_tot
                      in k_tot_dict.items()])
    k_escape_dict = dict([(expt, (1 - a_dict[expt]) * k_tot) for expt, k_tot
                          in k_tot_dict.items()])

    print(avg_free_vels_dict.values())
    print(f_oneway(*avg_free_vels_dict.values()))
    for expt, vels in avg_free_vels_dict.items():
        print('{}: {}'.format(expt, np.mean(vels)))
    print(f_oneway(*distances_dict.values()))
    for expt, distances in distances_dict.items():
        print('{}: {}'.format(expt, np.mean(distances)))
        print('{}: {}'.format(expt, np.median(distances)))
    print(np.nanmedian(np.concatenate(list(distances_dict.values()))))
    for expt, times in times_dict.items():
        print('{}: {}'.format(expt, np.mean(times)))

    print('k_on pm 95 CI')
    print('-------------')
    for expt, k_on in k_on_dict.items():
        print('{}: {} pm {}'.format(expt, k_on, 1.96 * k_on
                                    / np.sqrt(len(steps_dict[expt]))))

    print()
    print('k_es pm 95 CI')
    print('-------------')
    for expt, k_es in k_escape_dict.items():
        print('{}: {} pm {}'.format(expt, k_es, 1.96 * k_es
                                    / np.sqrt(len(steps_dict[expt]))))

    vels = np.concatenate(list(avg_free_vels_dict.values())) / Vstar
    scaled_times = np.concatenate(list(times_dict.values())) * Vstar / average_distance

    a_fit, a_full, e_fit, e_full, reduced = fit_fast_binding_params(
        scaled_times, vels)

    sim_dist = 2.5
    sim_dict = run_velocity_expts(Vstar, a_full, sim_dist, e_full * average_distance / sim_dist, k_off_dict, k_on_dict,
                                  k_escape_dict)
    sim_result = get_free_velocities_sim(sim_dict)
    (avg_free_vels_sim, distances_sim, times_sim, steps_sim, dwells_sim,
     ndwells_sim, vels_sim, escape_sim) = sim_result

    plot_avg_free_vels(avg_free_vels_dict, avg_free_vels_sim, colors)
    plot_free_distances(distances_dict, distances_sim, colors)
    plot_steps(steps_dict, steps_sim, k_on_dict, colors)
    plot_dwells(dwells_dict, dwells_sim, k_off_dict, colors)
    plot_escapes(escape_dict, escape_sim, colors)
    plot_ndwells(ndwells_dict, ndwells_sim, k_escape_dict, k_on_dict, colors)

    k_off_fast = (Vstar * a_full) / (average_distance * e_full)
    k_on_fast = (Vstar * (1 - a_full)) / (average_distance * e_full)
    print('k_off_fast = {}'.format(k_off_fast))
    print('k_on_fast = {}'.format(k_on_fast))
    plot_vels_and_fits(Vstar, a_fit, a_full, avg_free_vels_dict, e_fit, e_full, reduced, vels, colors)
    plot_vels_and_traj(experiments, sim_dict, vels_dict, vels_sim, k_on_dict, k_off_dict, average_distance, Vstar,
                       a_full, e_full, colors)
    # plot_vels_and_traj(experiments, time_slices, vels_dict, vels_slice, k_on_dict, k_off_dict, average_distance, Vstar,
                       # a_full, e_full, colors)
    # print(sol.x)


def slice_simulations(sim_dict, Vstar, slice_length=2.5):
    sliced_dict = dict()
    for expt, trajectories in sim_dict.items():
        sliced_trajectories = list()
        for trajectory in trajectories:
            slice = trajectory[trajectory[:, 1] < slice_length]
            t_end, y_end = slice[-1, :2]
            del_t = slice_length - y_end

            slice = np.concatenate(
                [slice, np.array([[del_t / Vstar + t_end, slice_length, 0]])],
                axis=0)

            sliced_trajectories.append(slice)
        sliced_dict.update([(expt, sliced_trajectories)])
    return sliced_dict


def run_velocity_expts(Vstar, a_full, average_distance, e_full, k_off_dict,
                       k_on_dict, k_escape_dict):
    assert k_off_dict.keys() == k_on_dict.keys()
    expts = k_off_dict.keys()

    sim_dict = {}
    for expt in expts:
        # Run velocity experiments
        k_on = k_on_dict[expt]
        k_off = k_off_dict[expt]
        k_escape = k_escape_dict[expt]
        c = k_off / (k_off + k_on)
        eps2 = (Vstar / average_distance) / (k_off + k_on)
        rate_escape = k_escape / (Vstar / average_distance)
        if np.isfinite(c):
            num_expts = 1024
            results = [
                experiment(a_full / e_full, (1 - a_full) / e_full, c / eps2, (1 - c) / eps2, rate_escape)
                for _ in range(num_expts)
            ]

            trajectories = [np.stack([res[1] * average_distance / Vstar,
                                      res[0] * average_distance, res[2]],
                                     axis=-1)
                            for res in results]
            # trunc_res = list()
            # for res in results:
            #     slow_bound = (res[2] == 2) + (res[2] == 3)
            #     bound_indices = np.nonzero(slow_bound)[0]
            #     if len(bound_indices) > 0:
            #         trunc_res.append(np.stack([
            #             res[1][bound_indices[0]:bound_indices[-1]]
            #             * average_distance / Vstar,
            #             res[0][bound_indices[0]:bound_indices[-1]]
            #             * average_distance
            #             ]
            #         ))
            sim_dict.update([(expt, trajectories)])
    return sim_dict


def plot_vels_and_fits(Vstar, a_fit, a_full, avg_free_vels_dict, e_fit, e_full,
                       reduced, vels, colors):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    s = ''
    for expt, averages in avg_free_vels_dict.items():
        ax[0].hist(averages, density=True, label=expt, alpha=0.7, color=colors[expt])
        x_cdf_plot = np.append(0, np.append(np.sort(averages), 52))
        y_cdf_plot = np.append(
            np.arange(averages.size + 1, dtype=float) / averages.size, 1)
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])
    v_plot = np.linspace(np.amin(vels), 1, num=257)
    # ax[0].plot(v_plot * Vstar, reduced(v_plot, a_fit, e_fit) / Vstar,
    #             label='Fit 1')
    ax[0].plot(v_plot * Vstar, reduced(v_plot, a_full, e_full) / Vstar,
               label='Model Fit')
    # ax[1].plot(
    #     v_plot * Vstar, 1 + cumtrapz(reduced(v_plot[::-1], a_fit, e_fit),
    #                                  v_plot[::-1], initial=0)[::-1]
    # )
    ax[1].plot(
        v_plot * Vstar, 1 + cumtrapz(reduced(v_plot[::-1], a_full, e_full),
                                     v_plot[::-1], initial=0)[::-1]
    )
    for a in ax:
        a.set_xlabel('Avg Free Velocity ($\\mu m / s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def fit_fast_binding_params(scaled_times, vels):
    def objective_simple(p, vels_temp):
        ap, epsp = p
        return -np.sum(np.log(q(1, 1. / vels_temp, ap, epsp)))

    def objective_two_variable(p, data):
        ap, epsp = p
        vels_temp, times_temp = data[:, 0], data[:, 1]
        return -np.sum(np.log(q(times_temp * vels_temp, times_temp, ap, epsp)))

    def q(y, s, a, eps):
        return (1 / np.sqrt(4 * np.pi * eps * a * (1 - a) * s)
                * (a + (y - a * s) / (2 * s))
                * np.exp(-(y - a * s) ** 2 / (4 * eps * a * (1 - a) * s)))

    def reduced(v, a_reduced, eps_reduced):
        b_reduced = 1 - a_reduced
        return (1 / np.sqrt(4 * np.pi * eps_reduced * a_reduced * b_reduced
                            * v ** 3) * (a_reduced + (v - a_reduced) / 2)
                * np.exp(-(v - a_reduced) ** 2
                         / (4 * eps_reduced * a_reduced * b_reduced * v)))

    v_bar = np.mean(vels)
    s2 = np.std(vels) ** 2
    a0 = v_bar - s2 / (2 * v_bar)
    e0 = s2 / (2 * v_bar ** 2 * (1 - v_bar))
    initial_guess = np.array([a0, e0])
    sol = minimize(objective_simple, initial_guess, args=(vels,),
                   bounds=[(0.05, .9), (0.01, None)])
    sol_full = minimize(objective_two_variable, initial_guess,
                        args=(np.stack([vels, scaled_times]),),
                        bounds=[(0.05, .9), (0.01, None)])
    a_fit, e_fit = sol.x
    a_full, e_full = sol_full.x
    return a_fit, a_full, e_fit, e_full, reduced


def plot_free_distances(distances_dict, distances_sim, colors):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    for expt, distances in distances_dict.items():
        ax[0].hist(distances, density=True, label=expt, alpha=0.7, color=colors[expt])
        x_cdf_plot = np.append(0, np.append(np.sort(distances), 65))
        y_cdf_plot = np.append(
            np.arange(distances.size + 1, dtype=float) / distances.size, 1)
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])

        try:
            dist_sim = distances_sim[expt]
            x_cdf_plot_sim = np.append(0, np.sort(dist_sim))
            y_cdf_plot_sim = (np.arange(dist_sim.size + 1, dtype=float)
                              / dist_sim.size)
            ax[1].step(x_cdf_plot_sim, y_cdf_plot_sim, linestyle='-.',
                       where='post', color=colors[expt])
        except KeyError:
            pass
    for a in ax:
        a.set_xlabel('Distance Traveled ($\\mu m$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def plot_avg_free_vels(avg_free_vels_dict, avg_free_vels_sim, colors):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    s = ''
    for expt, averages in avg_free_vels_dict.items():
        ax[0].hist(averages, density=True, label=expt, alpha=0.7, color=colors[expt])
        x_cdf_plot = np.append(0, np.append(np.sort(averages), 52))
        y_cdf_plot = np.append(
            np.arange(averages.size + 1, dtype=float) / averages.size, 1)
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])

        try:
            avg_sim = avg_free_vels_sim[expt]
            x_cdf_plot_sim = np.append(0, np.append(np.sort(avg_sim), 52))
            y_cdf_plot_sim = np.append(np.arange(avg_sim.size + 1, dtype=float)
                                       / avg_sim.size, 1)
            ax[1].step(x_cdf_plot_sim, y_cdf_plot_sim, linestyle='-.', where='post', color=colors[expt])
        except KeyError:
            pass
    for a in ax:
        a.set_xlabel('Avg Free Velocity ($\\mu m / s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def plot_steps(steps_dict, steps_sim, k_on_dict, colors):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    s = ''
    for expt, steps in steps_dict.items():
        ax[0].hist(steps,  density=True, label=expt, alpha=0.7, color=colors[expt])
        x_cdf_plot = np.append(0, np.sort(steps))
        y_cdf_plot = np.arange(steps.size + 1, dtype=float) / steps.size
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])

        # Fit steps
        k_on = k_on_dict[expt]
        err = k_on * 1.96 / np.sqrt(steps.size)
        s += '{} k_on = {:.2f} $\\pm$ {:.2f} $1/s$\n'.format(expt, k_on, err)
        # try:
        #     x_plot = np.linspace(0, np.amax(steps), num=200)
        #     y_plot = expon.pdf(x_plot, scale=1./k_on)
        #     ax[0].plot(x_plot, y_plot, '--', color=colors[expt], linewidth=2.)
        #     ax[1].plot(x_plot, expon.cdf(x_plot, scale=1./k_on), '--', color=colors[expt], linewidth=2.)
        # except ValueError:
        #     pass

        try:
            sim_step = steps_sim[expt]
            x_cdf_plot_sim = np.append(0, np.sort(sim_step))
            y_cdf_plot_sim = np.arange(sim_step.size + 1, dtype=float) / sim_step.size
            ax[1].step(x_cdf_plot_sim, y_cdf_plot_sim, linestyle='-.', where='post', color=colors[expt])
        except KeyError:
            pass
    for a in ax:
        a.set_xlabel('Step Time ($s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    ax[1].text(1.5, .4, s)
    plt.tight_layout()
    plt.show()


def plot_dwells(dwells_dict, dwells_sim, k_off_dict, colors):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    # s = ''
    for expt, dwells in dwells_dict.items():
        ax[0].hist(dwells,  density=True, label=expt, alpha=0.7, color=colors[expt])
        x_cdf_plot = np.append(0, np.sort(dwells))
        y_cdf_plot = np.arange(dwells.size + 1, dtype=float) / dwells.size
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])

        # k_off = k_off_dict[expt]
        # err = k_off * 1.96 / np.sqrt(dwells.size)
        # s += '{} k_off = {:.2f} $\\pm$ {:.2f} $1/s$\n'.format(expt, k_off, err)
        # x_plot = np.linspace(0, np.amax(dwells), num=200)
        # y_plot = expon.pdf(x_plot, scale=1./k_off)
        # ax[0].plot(x_plot, y_plot, '--', color=colors[expt], linewidth=2.)
        # ax[1].plot(x_plot, expon.cdf(x_plot, scale=1./k_off), '--', color=colors[expt], linewidth=2.)

        try:
            sim_dwell = dwells_sim[expt]
            x_cdf_plot_sim = np.append(0, np.sort(sim_dwell))
            y_cdf_plot_sim = np.arange(sim_dwell.size + 1, dtype=float) / sim_dwell.size
            ax[1].step(x_cdf_plot_sim, y_cdf_plot_sim, linestyle='-.', where='post', color=colors[expt])
        except KeyError:
            pass
    for a in ax:
        a.set_xlabel('Pause Time ($s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    # ax[1].text(30, .4, s)
    plt.tight_layout()
    plt.show()


def plot_escapes(escape_dict, escape_sim, colors):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    s = ''
    for expt, escape in escape_dict.items():
        ax[0].hist(escape, density=True, label=expt, alpha=0.7, color=colors[expt])
        x_cdf_plot = np.append(0, np.sort(escape))
        y_cdf_plot = np.arange(escape.size + 1, dtype=float) / escape.size
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])

        try:
            sim_escape = escape_sim[expt]
            x_cdf_plot_sim = np.append(0, np.sort(sim_escape))
            y_cdf_plot_sim = np.arange(sim_escape.size + 1, dtype=float) / sim_escape.size
            ax[1].step(x_cdf_plot_sim, y_cdf_plot_sim, linestyle='-.', where='post', color=colors[expt])
        except KeyError:
            pass
    for a in ax:
        a.set_xlabel('Escape time ($s$)')
    ax[0].set_ylabel('Probability density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    ax[1].text(2, .5, s)
    plt.tight_layout()
    plt.show()


def plot_ndwells(ndwells_dict, ndwells_sim, k_escape_dict, k_on_dict, colors):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    s = ''
    ndwell_max = max(np.amax(n_dwell) for n_dwell in ndwells_dict.values())
    bins = np.arange(0, ndwell_max+1) + 0.5
    for expt, ndwells in ndwells_dict.items():
        ax[0].hist(ndwells, density=True, bins=bins, label=expt, alpha=0.7,
                   color=colors[expt])
        # x_cdf_plot = np.append(0, np.sort(ndwells))
        # y_cdf_plot = np.arange(ndwells.size + 1, dtype=float) / ndwells.size
        # ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])
        ax[1].hist(ndwells, density=True, bins=bins, label=expt, alpha=0.7,
                   color=colors[expt], cumulative=True, histtype='step')

        mu = np.mean(ndwells)
        a = (mu - 1) / mu
        # a = k_on_dict[expt] / (k_on_dict[expt] + k_escape_dict[expt])
        x_plot = np.arange(1, 11)
        y_plot = a**(x_plot - 1) * (1 - a)
        ax[0].plot(x_plot, y_plot, '*', color=colors[expt])
        ax[1].plot(x_plot, np.cumsum(y_plot), '*', color=colors[expt])
        # err = k_off * 1.96 / np.sqrt(ndwells.size)
        # s += '{} k_off = {:.2f} $\\pm$ {:.2f} $1/s$\n'.format(expt, k_off, err)
        # x_plot = np.linspace(0, np.amax(ndwells), num=200)
        # y_plot = expon.pdf(x_plot, scale=1./k_off)
        # ax[0].plot(x_plot, y_plot, '--', color=colors[expt], linewidth=2.)
        # ax[1].plot(x_plot, expon.cdf(x_plot, scale=1./k_off), '--', color=colors[expt], linewidth=2.)

        # try:
        #     sim_ndwell = ndwells_sim[expt]
        #     # x_cdf_plot_sim = np.append(0, np.sort(sim_ndwell))
        #     # y_cdf_plot_sim = np.arange(sim_ndwell.size + 1, dtype=float) / sim_ndwell.size
        #     # ax[1].step(x_cdf_plot_sim, y_cdf_plot_sim, linestyle='-.', where='post', color=colors[expt])
        #     ax[0].hist(sim_ndwell, bins=bins, density=True, histtype='step',
        #                color=colors[expt], linestyle='--')
        #     ax[1].hist(sim_ndwell, bins=bins, density=True, cumulative=True,
        #                label='sim '+expt, alpha=0.7, histtype='step',
        #                color=colors[expt], linestyle='--')
        # except KeyError:
        #     pass
    for a in ax:
        a.set_xlabel('Number of dwells')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    # ax[1].text(30, .4, s)
    plt.tight_layout()
    plt.show()


def plot_vels_and_traj(experiments, simulations, vels_dict, vels_sim,
                       k_on_dict, k_off_dict, average_distance, Vstar, a, eps1,
                       colors):
    fig1, ax1 = plt.subplots(ncols=2, figsize=(10, 5))
    fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 5),
                             sharex='all', sharey='all')
    # fig3, ax3 = plt.subplots(ncols=2, figsize=(10, 5),
    #     #                          sharex='all', sharey='all')
    s1 = 'Data:\n'
    s2 = 'Model:\n'
    for expt, vels in vels_dict.items():
        ax1[0].hist(vels,  density=True, label=expt, alpha=0.7,
                    color=colors[expt])
        vels = vels[vels > 0]
        x_cdf_plot = np.append(0, np.sort(vels))
        y_cdf_plot = np.arange(vels.size + 1, dtype=float) / vels.size
        ax1[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[expt])

        mu = np.mean(vels)
        se = sem(vels)
        s1 += '{} mean vel = {:.2f} $\\pm$ {:.2f} \n'.format(expt, mu, se)

        # Truncate the trajectories for the experiment
        traj_list = experiments[expt]
        if expt[0] == 'h':
            color = 'b'
        else:
            color = 'r'
        for traj in traj_list:
            trunc_traj = truncate_trajectory(traj, absolute_pause_threshold=1.)
            if trunc_traj.shape[0] > 0 and np.all(
                    trunc_traj[1:, 0] - trunc_traj[:-1, 0] > 0):
                trunc_traj -= trunc_traj[0]
                if expt[-1] == 'w':
                    ax2[0].plot(trunc_traj[:, 0], trunc_traj[:, 1],
                                color=color, linewidth=0.4)
                # elif expt[-1] == 'p':
                #     ax3[0].plot(trunc_traj[:, 0], trunc_traj[:, 1],
                #                 color=color, linewidth=0.4)
                else:
                    raise ValueError('Problem with expt key')

        # # Run velocity experiments
        # k_on = k_on_dict[expt]
        # k_off = k_off_dict[expt]
        #
        # c = k_off / (k_off + k_on)
        # eps2 = (Vstar / average_distance) / (k_off + k_on)
        #
        # if np.isfinite(c):
        #     num_expts = 256
        #     results = [
        #         experiment(a / eps1, (1 - a) / eps1, c / eps2,  (1 - c) / eps2)
        #         for _ in range(num_expts)
        #     ]
        #
        #     expt_vels = list()
        #     for res in results:
        #         slow_bound = (res[2] == 2) + (res[2] == 3)
        #         if np.any(slow_bound):
        #             expt_vels.append(
        #                 (res[0][slow_bound][-1] - res[0][slow_bound][0])
        #                 / (res[1][slow_bound][-1] - res[1][slow_bound][0])
        #                 * Vstar)
        #
        #     expt_vels = np.array(expt_vels)
        #     # ax1[0].hist(expt_vels, density=True, label=expt, alpha=0.7,
        #     #            color=colors[c_counter])
        #     expt_vels = expt_vels[expt_vels > 0]
        try:
            sim_vel = vels_sim[expt]
            x_cdf_plot = np.append(0, np.sort(sim_vel))
            y_cdf_plot = (np.arange(sim_vel.size + 1, dtype=float)
                          / sim_vel.size)
            ax1[1].step(x_cdf_plot, y_cdf_plot, where='post', linestyle='--',
                        color=colors[expt])

            mewtwo = np.mean(sim_vel)
            se2 = sem(sim_vel)
            s2 += '{} mean vel = {:.2f} $\\pm$ {:.2f} \n'.format(
                expt, mewtwo, se2)

            if expt[0] == 'h':
                color = 'b'
            else:
                color = 'r'
        except KeyError:
            pass

        try:
            results = simulations[expt]
            for res in results:
                t, y, state = res.T
                bound_indices = np.nonzero((state == 2) + (state == 3))[0]

                if bound_indices.size > 0:
                    trunc_t = t[bound_indices[0]:bound_indices[-1] + 2]
                    trunc_y = y[bound_indices[0]:bound_indices[-1] + 2]
                    if expt[-1] == 'w':
                        ax2[1].plot(trunc_t,
                                    (trunc_y - trunc_y[0]),
                                    color=color, linewidth=0.4)
                    # elif expt[-1] == 'p':
                    #     ax3[1].plot(trunc_t,
                    #                 (trunc_y - trunc_y[0]),
                    #                 color=color, linewidth=0.4)
                    else:
                        raise ValueError('Problem with expt key')
        except KeyError:
            pass

        # x_plot = np.linspace(0, np.amax(vels), num=200)
        # y_plot = expon.pdf(x_plot, scale=1./k_off)
        # ax1[0].plot(x_plot, y_plot, '--', color=colors[c_counter], linewidth=2.)
        # ax1[1].plot(x_plot, expon.cdf(x_plot, scale=1./k_off), '--', color=colors[c_counter], linewidth=2.)

    for a in ax1:
        a.set_xlabel('Time Averaged Velocity ($\\mu m / s$)')
    for a in ax2:
        a.set_xlabel('Time ($s$)')
        a.set_ylabel('Displacement ($\\mu m$)')
    ax1[0].set_ylabel('Probability Density')
    ax1[1].set_ylabel('CDF')
    ax1[0].legend()
    ax1[1].text(6., .4, s1)
    ax1[1].text(6., .1, s2)
    ax1[1].set_xlim(right=17.5)

    ax2[0].set_title('Experiments')
    ax2[1].set_title('Simulations')

    fig1.tight_layout()
    fig2.tight_layout()
    # fig3.tight_layout()
    plt.show()


# def ecdf_plot(expt_data, sim_data, colors):
#     fig, ax = plt.subplots(ncols=2, sharex='all', figsize=(10, 5))
#     big_dataset = max(expt_data.values(), key=len)
#     bins = np.histogram(big_dataset, bins='auto')
#     for key, data in expt_data:
#         ax[0].hist(data, bins=bins, label=key, alpha=0.7, color=colors[key])
#
#         x_cdf_plot = np.append([0], np.sort(data))
#         y_cdf_plot = np.arange(data.size + 1, dtype=float) / data.size
#         ax[1].step(x_cdf_plot, y_cdf_plot, where='post', label=key,
#                    color=colors[key])
#
#         try:
#             sim = sim_data[key]
#
#
#             x_cdf_plot_sim = np.append(0, np.sort(sim))
#             y_cdf_plot_sim = np.arange(sim.size + 1, dtype=float) / sim.size
#             ax[1].step(x_cdf_plot_sim, y_cdf_plot_sim, where='post', label=key,
#                        linestyle='-.', color=colors[key])
#         except KeyError:
#             pass
#
#     return


def get_free_velocities(experiments, absolute_pause_threshold=0.):
    avg_free_vels_dict = dict()
    distances_dict = dict()
    times_dict = dict()
    steps_dict = dict()
    dwells_dict = dict()
    ndwells_dict = dict()
    vel_dict = dict()
    escape_dict = dict()
    for expt, trajectories in experiments.items():
        avg_free_vels_combined = list()
        distances_list = list()
        steps_list = list()
        dwell_list = list()
        ndwell_list = list()
        vel_list = list()
        escape_list = list()
        for trajectory in trajectories:
            y_save, t_save = process_trajectory(
                trajectory,
                absolute_pause_threshold=absolute_pause_threshold
            )[1:]
            if y_save.size == 2:
                continue
            result = extract_state_data(t_save, y_save)
            avg_free_vels, steps, dwells, free_vels, vels = (
                result[2], result[7], result[0], result[1], result[8])

            steps_list.append(steps / free_vels)
            dwell_list.append(dwells)
            ndwell_list.append(len(dwells))
            avg_free_vels_combined.append(avg_free_vels)
            distances_list.append(np.sum(steps))
            vel_list.append(vels)
            escape_list.append(np.sum(steps / free_vels))

        avg_free_vels_dict.update(
            [(expt, np.concatenate(avg_free_vels_combined))])
        distances_dict.update([(expt, np.array(
            [d for d in distances_list if d > 0]))])
        times_dict.update([(expt, np.array(
            [v / d for v, d in zip(avg_free_vels_dict[expt],
                                   distances_dict[expt])]))])
        steps_dict.update([(expt, np.concatenate(steps_list))])
        dwells_dict.update([(expt, np.concatenate(dwell_list))])
        ndwells_dict.update([(expt, np.array(ndwell_list))])
        vel_dict.update([(expt, np.array(vel_list))])
        escape_dict.update([(expt, np.array(escape_list))])
    return (avg_free_vels_dict, distances_dict, times_dict, steps_dict,
            dwells_dict, ndwells_dict, vel_dict, escape_dict)


def get_free_velocities_sim(experiments):
    avg_free_vels_dict = dict()
    distances_dict = dict()
    times_dict = dict()
    steps_dict = dict()
    dwells_dict = dict()
    ndwells_dict = dict()
    vel_dict = dict()
    escape_dict = dict()
    for expt, trajectories in experiments.items():
        avg_free_vels_combined = list()
        distances_list = list()
        steps_list = list()
        dwell_list = list()
        ndwell_list = list()
        vel_list = list()
        escape_list = list()
        for trajectory in trajectories:
            t, y, state = trajectory.T
            slow_bound = ((state == 2) + (state == 3))[:-1]
            slow_bound = np.append([False], slow_bound)
            # slow_bound[i] stores the bound state
            # in the interval t[i-1]--t[i]

            # Extract the transition times
            trans = slow_bound[1:] != slow_bound[:-1]
            trans = np.append(trans, True)
            trans[0] = True
            y_save = y[trans]
            t_save = t[trans]

            # assert len(y_save) % 2 == 0
            assert len(y_save) == len(t_save)

            if y_save.size == 2:
                continue

            result = extract_state_data(t_save, y_save)
            avg_free_vels, steps, dwells, free_vels, vels = (
                result[2], result[7], result[0], result[1], result[8])

            steps_list.append(steps / free_vels)
            dwell_list.append(dwells)
            ndwell_list.append(len(dwells))
            avg_free_vels_combined.append(avg_free_vels)
            distances_list.append(np.sum(steps))
            vel_list.append(vels)
            escape_list.append(np.sum(steps / free_vels))

        avg_free_vels_dict.update(
            [(expt, np.concatenate(avg_free_vels_combined))])
        distances_dict.update([(expt, np.array(
            [d for d in distances_list if d > 0]))])
        times_dict.update([(expt, np.array(
            [v / d for v, d in zip(avg_free_vels_dict[expt],
                                   distances_dict[expt])]))])
        steps_dict.update([(expt, np.concatenate(steps_list))])
        dwells_dict.update([(expt, np.concatenate(dwell_list))])
        ndwells_dict.update([(expt, np.array(ndwell_list))])
        vel_dict.update([(expt, np.array(vel_list))])
        escape_dict.update([(expt, np.array([escape_list]))])

    return (avg_free_vels_dict, distances_dict, times_dict, steps_dict,
            dwells_dict, ndwells_dict, vel_dict, escape_dict)


def fit_slow_binding_params(steps_dict, dwells_dict):
    k_on_dict = {}
    k_off_dict = {}
    for expt, steps in steps_dict.items():
        # Fit steps
        k_on = 1 / np.mean(steps)
        k_on_dict.update([(expt, k_on)])

    for expt, dwells in dwells_dict.items():
        # Fit dwells
        k_off = 1 / np.mean(dwells)
        k_off_dict.update([(expt, k_off)])

    return k_on_dict, k_off_dict


if __name__ == '__main__':
    main()
