import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from free_velocity_processing import fit_fast_binding_params, get_free_velocities
from extract_trajectory_data import load_trajectories, extract_state_data, process_trajectory
from jv import solve_pde, delta_h
from mle import change_vars


if __name__ == '__main__':
    collagen = ['hcp', 'ccp', 'hcw', 'ccw']
    fibrinogen = ['hfp', 'ffp', 'hfw', 'ffw']
    vWF = ['hvp', 'vvp']
    experiments = load_trajectories(fibrinogen)
    avg_free_vels_dict, distances_dict, times_dict = get_free_velocities(experiments)

    Vstar = 50
    vels = np.concatenate(avg_free_vels_dict.values()) / Vstar
    scaled_times = np.concatenate(times_dict.values()) / 8.5  # Where did this magic parameter come from?

    a_fit, a_full, e_fit, e_full, reduced = fit_fast_binding_params(
        scaled_times, vels)

    # def full_objective(p, v, vmin, N_obj):
    #     c, eps2 = change_vars(p, forward=False)[:, 0]
    #
    #     h = 1. / N_obj
    #     s_eval = (np.arange(0, np.ceil(1. / (vmin * h))) + 1) * h
    #
    #     y = np.linspace(0, 1, num=N_obj + 1)
    #     u_init = delta_h(y[1:], h)
    #     p0 = np.append(u_init, np.zeros(4 * N_obj))
    #     u1_bdy = solve_pde(s_eval, p0, h, eps1=e_full, eps2=eps2, a=a_full,
    #                        b=1 - a_full, c=c, d=1-c, scheme='up')[3]
    #
    #     pdf = np.interp(1. / v, s_eval,
    #                     u1_bdy / (1 - np.exp((a_full - 1) / e_full + (c - 1) / eps2)))
    #     return -np.sum(np.log(pdf))
    #
    # N_obj = 32
    # sol_combined = dict()
    # for expt, trajectories in experiments.items():
    #     v_combined = list()
    #     for trajectory in experiments[expt]:
    #         y_save, t_save = process_trajectory(trajectory)[1:]
    #         v_combined.append(extract_state_data(t_save, y_save)[8])
    #
    #     v_combined = np.array(v_combined)
    #     v_combined = v_combined[v_combined > 0] / Vstar
    #
    #     vmin = np.amin(v_combined)
    #     initial_guess = change_vars(np.array([0.5, 1]), forward=True)
    #     sol = minimize(full_objective, initial_guess, args=(v_combined, vmin, N_obj))
    #     sol_combined.update([(expt, sol)])

    for expt, trajectories in experiments.items():
        v_combined = list()
        step_combined = list()
        dwell_combined = list()
        ndwell_combined = list()
        for trajectory in trajectories:
            y_save, t_save = process_trajectory(trajectory)[1:]
            data = extract_state_data(t_save, y_save)
            d, s, v = data[0], data[7], data[8]
            dwell_combined.append(d)
            step_combined.append(s)
            v_combined.append(v)
            ndwell_combined.append(len(d))

        dwell_combined = np.concatenate(dwell_combined)  # / 8.5  # Where did this magic parameter come from?
        ndwell_combined = np.array(ndwell_combined)

        v_combined = np.array(v_combined)
        v_combined = v_combined[v_combined > 0] / Vstar

        step_combined = np.concatenate(step_combined) / Vstar  # / (8.5 * Vstar)  # Magic parameter 8.5

        k_off = 1 / np.mean(dwell_combined)
        k_on = 1 / np.mean(step_combined)

        print('{}: {}, {}, {}'.format(expt, k_off, k_on, np.mean(v_combined)))
        print(np.mean(ndwell_combined))
        print(np.amin(v_combined))



    # print(change_vars(sol.x, forward=False))
