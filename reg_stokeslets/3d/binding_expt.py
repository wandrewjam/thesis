import numpy as np
from motion_integration import integrate_motion, nondimensionalize, eps_picker
from resistance_matrix_test import generate_resistance_matrices
import matplotlib.pyplot as plt


if __name__ == '__main__':
    expt = 2
    t_span = [0., 8.]
    num_steps = 200

    def exact_vels(em):
        return np.zeros(6)

    n_nodes = 8
    a, b = 1.5, .5
    domain = 'wall'
    adaptive = False

    if expt == 1:
        bonds = np.array([[0, 0., 0.]])
        init = np.array([.6, 0, 0, 1., 0, 0])
        receptors = np.array([[-.5, 0., 0.]])
    elif expt == 2:
        bonds = np.array([[0, 0., 0.]])
        init = np.array([1.55, 0, 0, 0, 0, -1.])
        receptors = np.array([[0., 1.5, 0.]])
    elif expt == 3:
        init = np.array([1.55, 0, 0, 0, 0, -1.])
        receptors = np.load('Xb_0.26.npy')
        bonds = np.zeros(shape=(0, 3), dtype='float')
        # bonds = np.array([[np.argmax(receptors[:, 1]), 0., 0.]])
    else:
        raise ValueError('expt is not valid')

    t_sc, f_sc, lam, k0_on, k0_off, eta, eta_ts, kappa = nondimensionalize(
        l_scale=1, shear=100., mu=4e-3, l_sep=0., dimk0_on=1000000., dimk0_off=5.,
        sig=100., sig_ts=99., temp=310.)

    result = integrate_motion(
        t_span, num_steps, init, exact_vels, n_nodes, a, b, domain,
        adaptive=adaptive, receptors=receptors, bonds=bonds, eta=eta,
        eta_ts=eta_ts, kappa=kappa, lam=lam, k0_on=k0_on, k0_off=k0_off)

    t = np.linspace(t_span[0], t_span[1], num_steps + 1) * t_sc
    mask = result[0] > 0
    plt.plot(t[mask], result[0][mask], t[mask], result[2][mask])
    plt.xlabel('Time (s)')
    plt.ylabel('Center of mass (\\mu m)')
    plt.legend(['x', 'z'])
    plt.show()

    plt.plot(t[mask], result[3][2, 0, mask])
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation component')
    plt.legend(['e_mz'])
    plt.show()

    center = np.stack(result[:3])
    if expt == 3:
        lengths = np.linalg.norm(center + np.dot(result[3].transpose(
            (2, 0, 1)), receptors[np.argmax(receptors[:, 1]), :].T).T, axis=0)
    else:
        lengths = np.linalg.norm(
            center[:, None, :] + np.dot(result[3].transpose((2, 0, 1)),
                                        receptors.T).transpose((1, 2, 0)),
            axis=0)
    plt.plot(t[mask], lengths.T[mask])
    plt.title('Bond length')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (\\mu m)')
    plt.show()

    bond_num = [bond.shape[0] for bond in result[-1]]
    print(bond_num)

    theta = np.arctan2(result[3][2, 0, mask][-1], result[3][1, 0, mask][-1])
    phi = np.arccos(result[3][0, 0, mask][-1])

    eps = eps_picker(n_nodes, a=a, b=b)
    shear_f, shear_t = generate_resistance_matrices(
        eps, n_nodes, a=a, b=b, domain=domain, distance=center[0, mask][-1],
        theta=theta, phi=phi)[-2:]

    print(np.stack([shear_f, shear_t]))

    print('Done!')
