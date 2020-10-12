import numpy as np
from scipy.linalg import solve
from timeit import default_timer as timer
from motion_integration import eps_picker
from force_test import assemble_quad_matrix
from resistance_matrix_test import assemble_vel_cases


if __name__ == '__main__':
    n_nodes = 24
    eps = eps_picker(n_nodes, a=1.5, b=.5)

    s_matrix, weights, nodes = assemble_quad_matrix(
        eps, n_nodes, 1.5, .5, 'wall', 1.5, np.pi/2, np.pi/4)

    rhs = assemble_vel_cases(nodes, distance=1.5, shear_vec=True)
    rhs_cases = rhs.shape[1]

    start = timer()
    intermediate_solve = solve(
        s_matrix, rhs, overwrite_a=True, overwrite_b=True, check_finite=False,
        assume_a='pos')

    pt_forces = (intermediate_solve.T / np.repeat(weights, repeats=3)).T
    pt_forces = pt_forces.reshape((-1, 3, rhs_cases))
    end = timer()

    print(end - start)
