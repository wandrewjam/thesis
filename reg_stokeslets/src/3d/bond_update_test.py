import numpy as np
from motion_integration import update_bonds


if __name__ == '__main__':
    bond_history = []
    N = 10
    receptors = np.stack([np.ones(N), np.random.rand(N), np.random.rand(N)],
                         axis=1)
    bonds = np.zeros(shape=(0, 3))
    bond_history.append(bonds)
    dt = .1
    k0_on, k0_off, eta = 1, 1, 1

    for i in range(100):
        bonds = update_bonds(receptors, bonds, 0, 0, 0, np.eye(3), dt, k0_on, k0_off, eta, eta_ts, 0)
        bond_history.append(bonds)

    n_bonds = np.zeros(101)
    for i, el in enumerate(bond_history):
        n_bonds[i] = el.shape[0]

    print(n_bonds)
