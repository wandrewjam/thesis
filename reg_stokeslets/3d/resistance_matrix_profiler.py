import cProfile
from resistance_matrix_test import generate_resistance_matrices
from motion_integration import eps_picker


def main(n_nodes):
    eps = eps_picker(n_nodes, a=1.5, b=0.5)
    generate_resistance_matrices(eps, n_nodes, a=1.5, b=0.5, domain='wall',
                                 distance=1.2)


if __name__ == '__main__':
    cProfile.run('main(8)', 'prof8.stats')
    cProfile.run('main(16)', 'prof16.stats')
    cProfile.run('main(24)', 'prof24.stats')
    cProfile.run('main(32)', 'prof32.stats')
