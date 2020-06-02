import numpy as np
from force_test import assemble_quad_matrix
from timeit import default_timer as timer
from sphere_integration_utils import compute_helper_funs


def main():
    n_nodes = 24
    eps = .01

    r_save = np.linspace(0, 8, num=10**4)
    precompute_array = (r_save, ) + tuple(compute_helper_funs(r_save, eps))

    start = timer()
    assemble_quad_matrix(eps, n_nodes, domain="wall", distance=2., proc=1)
    end = timer()
    print('Single threaded: {} sec'.format(end - start))

    start = timer()
    assemble_quad_matrix(eps, n_nodes, domain="wall", distance=2., proc=16)
    end = timer()
    print('Multi threaded: {} sec'.format(end - start))

    start = timer()
    assemble_quad_matrix(eps, n_nodes, domain="wall", distance=2., proc=1,
                         precompute_array=precompute_array)
    end = timer()
    print('Single threaded, precompute: {} sec'.format(end - start))

    start = timer()
    assemble_quad_matrix(eps, n_nodes, domain="wall", distance=2., proc=16,
                         precompute_array=precompute_array)
    end = timer()
    print('Multi threaded, precompute: {} sec'.format(end - start))


if __name__ == '__main__':
    main()
