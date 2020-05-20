from force_test import assemble_quad_matrix
from timeit import default_timer as timer


def main():
    n_nodes = 12
    eps = .01

    start = timer()
    assemble_quad_matrix(eps, n_nodes, domain="wall", distance=2., proc=1)
    end = timer()
    print('Single threaded: {} sec'.format(end - start))

    start = timer()
    assemble_quad_matrix(eps, n_nodes, domain="wall", distance=2., proc=3)
    end = timer()
    print('Multi threaded: {} sec'.format(end - start))


if __name__ == '__main__':
    main()
