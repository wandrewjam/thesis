import numpy as np


def main():
    eps = 0.1
    n_nodes = 2**3
    generate_resistance_matrices(eps, n_nodes)


def generate_resistance_matrices(eps, n_nodes):
    print('Assembling quadrature matrix for eps = {}, nodes = {}'.format(
        eps, n_nodes))

    # a_matrix, nodes = assemble_quad_matrix(eps=eps, n_nodes=n_nodes, a=a, b=b)


if __name__ == '__main__':
    main()
