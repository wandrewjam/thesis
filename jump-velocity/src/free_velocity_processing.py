import numpy as np
from scipy.stats import f_oneway, kruskal, expon, sem
from scipy.optimize import minimize
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
# from matplotlib import cm
from extract_trajectory_data import (extract_state_data, load_trajectories,
                                     process_trajectory)
from simulate import experiment


def main():
    collagen = ['hcp', 'ccp', 'hcw', 'ccw']
    fibrinogen = ['hfp', 'ffp', 'hfw', 'ffw']
    vWF = ['hvp', 'vvp']
    experiments = load_trajectories(fibrinogen)
    result = get_free_velocities(experiments, absolute_pause_threshold=1.)
    (avg_free_vels_dict, distances_dict, times_dict, steps_dict, dwells_dict,
     ndwells_dict, vels_dict) = result

    Vstar = 50
    average_distance = 2.5

    plot_avg_free_vels(avg_free_vels_dict)
    plot_free_distances(distances_dict)
    k_on_dict = plot_steps(steps_dict)
    k_off_dict = plot_dwells(dwells_dict)

    print(avg_free_vels_dict.values())
    print(f_oneway(*avg_free_vels_dict.values()))
    for expt, vels in avg_free_vels_dict.items():
        print('{}: {}'.format(expt, np.mean(vels)))
    print(f_oneway(*distances_dict.values()))
    for expt, distances in distances_dict.items():
        print('{}: {}'.format(expt, np.mean(distances)))
        print('{}: {}'.format(expt, np.median(distances)))
    print(np.nanmedian(np.concatenate(distances_dict.values())))
    for expt, times in times_dict.items():
        print('{}: {}'.format(expt, np.mean(times)))

    vels = np.concatenate(avg_free_vels_dict.values()) / Vstar
    scaled_times = np.concatenate(times_dict.values()) * Vstar / average_distance

    a_fit, a_full, e_fit, e_full, reduced = fit_fast_binding_params(
        scaled_times, vels)
    k_off = (Vstar * a_full) / (average_distance * e_full)
    k_on = (Vstar * (1 - a_full)) / (average_distance * e_full)
    print(k_off, k_on)
    plot_vels_and_fits(Vstar, a_fit, a_full, avg_free_vels_dict, e_fit, e_full,
                       reduced, vels)
    plot_vels(vels_dict, k_on_dict, k_off_dict, average_distance, Vstar,
              a_full, e_full)
    # print(sol.x)


def plot_vels_and_fits(Vstar, a_fit, a_full, avg_free_vels_dict, e_fit, e_full,
                       reduced, vels):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    colors = ['r', 'b', 'g', 'k']
    c_counter = 0
    s = ''
    for expt, averages in avg_free_vels_dict.items():
        ax[0].hist(averages, density=True, label=expt, alpha=0.7, color=colors[c_counter])
        x_cdf_plot = np.append(0, np.append(np.sort(averages), 52))
        y_cdf_plot = np.append(
            np.arange(averages.size + 1, dtype=float) / averages.size, 1)
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[c_counter])
        c_counter += 1
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
        return -np.sum(np.log(q(times_temp, 1. / vels_temp, ap, epsp)))

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


def plot_free_distances(distances_dict):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    colors = ['r', 'b', 'g', 'k']
    c_counter = 0
    for expt, distances in distances_dict.items():
        ax[0].hist(distances, density=True, label=expt, alpha=0.7, color=colors[c_counter])
        x_cdf_plot = np.append(0, np.append(np.sort(distances), 65))
        y_cdf_plot = np.append(
            np.arange(distances.size + 1, dtype=float) / distances.size, 1)
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[c_counter])
        c_counter += 1
    for a in ax:
        a.set_xlabel('Distance Traveled ($\\mu m$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def plot_avg_free_vels(avg_free_vels_dict):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    colors = ['r', 'b', 'g', 'k']
    c_counter = 0
    s = ''
    for expt, averages in avg_free_vels_dict.items():
        ax[0].hist(averages, density=True, label=expt, alpha=0.7, color=colors[c_counter])
        x_cdf_plot = np.append(0, np.append(np.sort(averages), 52))
        y_cdf_plot = np.append(
            np.arange(averages.size + 1, dtype=float) / averages.size, 1)
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[c_counter])
        c_counter += 1
    for a in ax:
        a.set_xlabel('Avg Free Velocity ($\\mu m / s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def plot_steps(steps_dict):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    colors = ['r', 'b', 'g', 'k']
    c_counter = 0
    s = ''
    k_on_dict = {}
    for expt, steps in steps_dict.items():
        ax[0].hist(steps,  density=True, label=expt, alpha=0.7, color=colors[c_counter])
        x_cdf_plot = np.append(0, np.sort(steps))
        y_cdf_plot = np.arange(steps.size + 1, dtype=float) / steps.size
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[c_counter])

        # Fit steps
        k_on = 1 / np.mean(steps)
        k_on_dict.update([(expt, k_on)])
        err = k_on * 1.96 / np.sqrt(steps.size)
        s += '{} k_on = {:.2f} $\\pm$ {:.2f} $1/s$\n'.format(expt, k_on, err)
        try:
            x_plot = np.linspace(0, np.amax(steps), num=200)
            y_plot = expon.pdf(x_plot, scale=1./k_on)
            ax[0].plot(x_plot, y_plot, '--', color=colors[c_counter], linewidth=2.)
            ax[1].plot(x_plot, expon.cdf(x_plot, scale=1./k_on), '--', color=colors[c_counter], linewidth=2.)
        except ValueError:
            pass
        c_counter += 1
    for a in ax:
        a.set_xlabel('Step Time ($s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    ax[1].text(.3, .4, s)
    plt.tight_layout()
    plt.show()
    return k_on_dict


def plot_dwells(dwells_dict):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    colors = ['r', 'b', 'g', 'k']
    c_counter = 0
    s = ''
    k_off_dict = {}
    for expt, dwells in dwells_dict.items():
        ax[0].hist(dwells,  density=True, label=expt, alpha=0.7, color=colors[c_counter])
        x_cdf_plot = np.append(0, np.sort(dwells))
        y_cdf_plot = np.arange(dwells.size + 1, dtype=float) / dwells.size
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post', color=colors[c_counter])

        # Fit steps
        k_off = 1 / np.mean(dwells)
        k_off_dict.update([(expt, k_off)])
        err = k_off * 1.96 / np.sqrt(dwells.size)
        s += '{} k_off = {:.2f} $\\pm$ {:.2f} $1/s$\n'.format(expt, k_off, err)
        x_plot = np.linspace(0, np.amax(dwells), num=200)
        y_plot = expon.pdf(x_plot, scale=1./k_off)
        ax[0].plot(x_plot, y_plot, '--', color=colors[c_counter], linewidth=2.)
        ax[1].plot(x_plot, expon.cdf(x_plot, scale=1./k_off), '--', color=colors[c_counter], linewidth=2.)
        c_counter += 1
    for a in ax:
        a.set_xlabel('Pause Time ($s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    ax[1].text(8, .4, s)
    plt.tight_layout()
    plt.show()
    return k_off_dict


def plot_vels(vels_dict, k_on_dict, k_off_dict, average_distance,
              Vstar, a, eps1):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    colors = ['r', 'b', 'g', 'k']
    c_counter = 0
    s1 = 'Data:\n'
    s2 = 'Model:\n'
    for expt, vels in vels_dict.items():
        ax[0].hist(vels,  density=True, label=expt, alpha=0.7,
                   color=colors[c_counter])
        vels = vels[vels > 0]
        x_cdf_plot = np.append(0, np.sort(vels))
        y_cdf_plot = np.arange(vels.size + 1, dtype=float) / vels.size
        ax[1].step(x_cdf_plot, y_cdf_plot, where='post',
                   color=colors[c_counter])

        mu = np.mean(vels)
        se = sem(vels)
        s1 += '{} mean vel = {:.2f} $\\pm$ {:.2f} \n'.format(expt, mu, se)

        # Run velocity experiments
        k_on = k_on_dict[expt]
        k_off = k_off_dict[expt]

        c = k_off / (k_off + k_on)
        eps2 = (Vstar / average_distance) / (k_off + k_on)

        if np.isfinite(c):
            num_expts = 256
            results = [
                experiment(a / eps1, (1 - a) / eps1, c / eps2,  (1 - c) / eps2)
                for _ in range(num_expts)
            ]

            expt_vels = list()
            for res in results:
                expt_vels.append((res[0][-2] - res[0][1])
                                 / (res[1][-2] - res[1][1]) * Vstar)

            expt_vels = np.array(expt_vels)
            # ax[0].hist(expt_vels, density=True, label=expt, alpha=0.7,
            #            color=colors[c_counter])
            expt_vels = expt_vels[expt_vels > 0]
            x_cdf_plot = np.append(0, np.sort(expt_vels))
            y_cdf_plot = (np.arange(expt_vels.size + 1, dtype=float)
                          / expt_vels.size)
            ax[1].step(x_cdf_plot, y_cdf_plot, where='post', linestyle='--',
                       color=colors[c_counter])

            mewtwo = np.mean(expt_vels)
            se2 = sem(expt_vels)
            s2 += '{} mean vel = {:.2f} $\\pm$ {:.2f} \n'.format(expt, mewtwo,
                                                                 se2)
        # x_plot = np.linspace(0, np.amax(vels), num=200)
        # y_plot = expon.pdf(x_plot, scale=1./k_off)
        # ax[0].plot(x_plot, y_plot, '--', color=colors[c_counter], linewidth=2.)
        # ax[1].plot(x_plot, expon.cdf(x_plot, scale=1./k_off), '--', color=colors[c_counter], linewidth=2.)
        c_counter += 1
    for a in ax:
        a.set_xlabel('Time Averaged Velocity ($\\mu m / s$)')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('CDF')
    ax[0].legend()
    ax[1].text(6., .4, s1)
    ax[1].text(6., .1, s2)
    ax[1].set_xlim(right=17.5)
    plt.tight_layout()
    plt.show()


def get_free_velocities(experiments, absolute_pause_threshold=0.):
    avg_free_vels_dict = dict()
    distances_dict = dict()
    times_dict = dict()
    steps_dict = dict()
    dwells_dict = dict()
    ndwells_dict = dict()
    vel_dict = dict()
    for expt, trajectories in experiments.items():
        avg_free_vels_combined = list()
        distances_list = list()
        steps_list = list()
        dwell_list = list()
        ndwell_list = list()
        vel_list = list()
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
    return (avg_free_vels_dict, distances_dict, times_dict, steps_dict,
            dwells_dict, ndwells_dict, vel_dict)


if __name__ == '__main__':
    main()
