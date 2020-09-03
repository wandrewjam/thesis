import numpy as np
import matplotlib.pyplot as plt
from motion_integration import integrate_motion, evaluate_motion_equations


def run_experiment(dist=0., a=1., b=1., exact_vels=None, proc=1,
                   forces=None, torques=None):
    stop = 50.
    t_steps = 100
    init = np.zeros(6)
    order = '2nd'

    if exact_vels is None:
        def exact_vels(em):
            return np.zeros(6)

    n_nodes = 8
    if dist == 0.:
        dom = 'free'
    elif dist >= 0:
        dom = 'wall'
    else:
        raise ValueError('dist is an unexpected value')

    init[0] = dist
    init[3] = 1.

    result = integrate_motion(
        [0., stop], t_steps, init, exact_vels, n_nodes, a=a, b=b,
        domain=dom, order=order, adaptive=False, proc=proc, forces=forces,
        torques=torques)

    x1, x2, x3, e_m = result[:4]

    return x1, x2, x3, e_m


def find_exact_solution(f, dist):
    stop = 50.
    t_steps = 100
    init = np.zeros(6)
    init[0] = dist
    init[3] = 1.

    t = np.linspace(0, stop, num=t_steps + 1)
    dt = t[1] - t[0]

    x_vec = np.zeros(shape=(3, t_steps + 1))
    em_fine = np.zeros(shape=(3, t_steps + 1))
    x_vec[:, 0] = init[:3]
    em_fine[:, 0] = init[3:]

    for (i, t_el) in enumerate(t[1:]):
        rhs = f(em_fine[:, i])
        d_x = rhs[:3]
        d_em = rhs[3:]

        cur_e = np.cross(d_em, em_fine[:, i])
        tmp_e = (em_fine[:, i] + dt * cur_e)
        tmp_e /= np.linalg.norm(tmp_e)

        prd_p = f(tmp_e)[:3]
        prd_e = np.cross(f(tmp_e)[3:], tmp_e)

        x_vec[:, i+1] = x_vec[:, i] + dt / 2 * (d_x + prd_p)
        em_fine[:, i+1] = (em_fine[:, i] + dt / 2 * (cur_e + prd_e))
        em_fine[:, i+1] /= np.linalg.norm(em_fine[:, i+1])

    return x_vec, em_fine, t


def main():
    # Construct experiment list
    expt_list = list()

    evaluate_motion_equations.counter = 0

    for (a, b) in [(1., 1.), (1.5, .5)]:
        for gen_forces in np.eye(6)[1:]:
            forces = gen_forces[:3]
            torques = gen_forces[3:]
            if a == 1.:
                dist = 0
                trn = -forces / (6 * np.pi * a)
                rot = -np.array([0, .5, 0]) - torques / (4 * np.pi * a**3)

                def exact_vels(em):
                    return np.concatenate([trn, rot])

                x_vec, em_fine, t = find_exact_solution(
                    exact_vels, dist)
                expt_list.append(
                    {'a': a, 'b': b, 'dist': dist,
                     'forces': forces, 'torques': torques,
                     'x_vec': x_vec, 'em_fine': em_fine, 't': t}
                    )
            elif a == 1.5:
                dist = 0.

                e = np.sqrt(1 - b**2 / a**2)

                xa = 8. / 3 * e**3 * (
                        2 * (2*e**2 - 1) * np.arctan(e / np.sqrt(1 - e**2))
                + 2 * e * np.sqrt(1 - e**2)) ** (-1)
                ya = 8. / 3 * e**3 * (
                        (2*e**2 + 1) * np.arctan(e / np.sqrt(1 - e**2))
                - e * np.sqrt(1 - e**2)) ** (-1)

                xc = 2. / 3 * e**3 * (np.arctan(e / np.sqrt(1 - e**2))
                                      - e * np.sqrt(1 - e**2)) ** (-1)
                yc = 2. / 3 * e**3 * (2 - e**2) * (
                        e * np.sqrt(1 - e**2) - (1 - 2*e**2)
                        * np.arctan(e / np.sqrt(1 - e**2))
                ) ** (-1)
                yh = 2. / 3 * e**5 * (
                        e * np.sqrt(1 - e**2) - (1 - 2 * e**2)
                        * np.arctan(e / np.sqrt(1 - e**2))
                ) ** (-1)
                eps = np.zeros(shape=(3, 3, 3))
                eps[0, 1, 2], eps[1, 2, 0], eps[2, 0, 1] = 1, 1, 1
                eps[0, 2, 1], eps[2, 1, 0], eps[1, 0, 2] = -1, -1, -1
                ros = np.zeros(shape=(3, 3))
                ros[0, 2], ros[2, 0] = 0.5, 0.5
                om_inf = np.array([0, -.5, 0])

                def exact_vels(em):
                    I = np.eye(3)
                    outer = em[:, None] * em[None, :]
                    Aa = xa * outer + ya * (I - outer)
                    Ac = xc * outer + yc * (I - outer)

                    rhs_f = -forces / (6 * np.pi * a)
                    rhs_t = yh / 2 * np.tensordot(np.dot(
                        (eps[:, :, None, :] * em[None, None, :, None]
                         + eps[:, None, :, :] * em[None, :, None, None]), em
                    ), ros) + np.dot(Ac, om_inf) - torques / (8 * np.pi * a**3)
                    trn_vels = np.linalg.solve(Aa, rhs_f)
                    rot_vels = np.linalg.solve(Ac, rhs_t)
                    return np.concatenate((trn_vels, rot_vels))

                x_vec, em_fine, t = find_exact_solution(
                    exact_vels, dist)
                expt_list.append(
                    {'a': a, 'b': b, 'dist': dist,
                     'forces': forces, 'torques': torques,
                     'x_vec': x_vec, 'em_fine': em_fine, 't': t}
                    )

    for (i, expt) in enumerate(expt_list):
        a, b, dist = expt['a'], expt['b'], expt['dist']
        f1, f2, f3 = expt['forces']
        t1, t2, t3 = expt['torques']
        x_vec, em_fine, t = expt.pop('x_vec'), expt.pop('em_fine'), expt.pop('t')
        x1, x2, x3, e_m = run_experiment(**expt)

        print('a = {}, b = {}, dist = {}, forces = ({}, {}, {}), '
              'torques = ({}, {}, {})'.format(a, b, dist, f1, f2,
                                              f3, t1, t2, t3))

        fig, ax = plt.subplots(2, 2, sharex='all', figsize=(12, 9))

        ax[0,0].plot(t, x2, t, x3, t, x_vec[1], t, x_vec[2])
        ax[0,0].legend(['$y$ approx', '$z$ approx', '$y$ analytic',
                        '$z$ analytic'])
        ax[0,0].set_ylabel('CoM position')

        ax[0,1].plot(t, e_m[0], t, e_m[1], t, e_m[2], t, em_fine[0], t, em_fine[1],
                 t, em_fine[2])
        ax[0,1].legend(['$e_{mx}$ approx', '$e_{my}$ approx', '$e_{mz}$ approx',
                        '$e_{mx}$ analytic', '$e_{my}$ analytic',
                        '$e_{mz}$ analytic'])
        ax[0,1].set_ylabel('Components of orientation vector')

        ax[1,0].plot(t, x2 - x_vec[1], t, x3 - x_vec[2])
        ax[1,0].legend(['$y$ error', '$z$ error'])
        ax[1,0].set_xlabel('t')
        ax[1,0].set_ylabel('CoM error')

        ax[1,1].plot(t, e_m[0] - em_fine[0], t, e_m[1] - em_fine[1],
                     t, e_m[2] - em_fine[2])
        ax[1,1].legend(['$e_{mx}$ error', '$e_{my}$ error', '$e_{mz}$ error'])
        ax[1,1].set_xlabel('t')
        ax[1,1].set_ylabel('Orientation error')

        # plt.savefig('f_test{}'.format(i), bbox_inches='tight')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
