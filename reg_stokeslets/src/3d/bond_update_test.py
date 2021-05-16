import numpy as np
from motion_integration import update_bonds


if __name__ == '__main__':
    bond_history = []
    N = 10
    receptors = np.stack([np.ones(N), np.random.rand(N), np.random.rand(N)],
                         axis=1)
    bonds = np.zeros(shape=(0, 4))
    bond_history.append(bonds)
    dt = .1
    k0_on, k0_off, eta, eta_ts = .01, 1, 1, 0.1
    k0_on2 = 1
    rng = np.random.RandomState(1000)

    for i in range(100):
        bonds = update_bonds(receptors, bonds, 0, 0, 0, np.eye(3), dt, k0_on,
                             k0_off, k0_on2, eta, eta_ts, 1, 0.1, rng=rng)[0]
        bond_history.append(bonds)

    n_bonds = np.zeros(101)
    for i, el in enumerate(bond_history):
        n_bonds[i] = el[el[:, -1] == 2].shape[0]

    print(n_bonds)
