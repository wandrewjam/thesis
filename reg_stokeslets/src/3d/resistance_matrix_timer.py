from resistance_matrix_test import generate_resistance_matrices
from timeit import default_timer as timer


def main(server):
    if server == 'linux':
        eps = [0.1, 0.05, 0.01]
        n_nodes = [4, 12, 24, 36]
    elif server == 'mac':
        eps = [0.1, 0.05, 0.01]
        n_nodes = [4, 8, 12]
    else:
        raise ValueError('\'server\' variable is not valid')

    for n in n_nodes:
        start = timer()
        for e in eps:
            generate_resistance_matrices(e, n, a=1.5, b=.5, domain='wall',
                                         distance=2.0, shear_vec=True)
        end = timer()
        print('For n = {}, generating the resistance matrix took {} sec on '
              'average'.format(n, (end-start)/3))


if __name__ == '__main__':
    import sys

    main(server=sys.argv[1])
