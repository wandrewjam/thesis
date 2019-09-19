import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from jv import solve_pde, delta_h


def solve_reduced(s_eval, p0, h, eps1, eps2, a, b, c, d, s_samp=1024):
    u00, u10, v0, f0, vf0 = np.split(p0, 5)
    v10, v20 = u00 + v0, f0 + vf0
    N = u00.shape[0]
    yi = np.linspace(0, 1, num=N+1)
    ystore = np.linspace(0, N, num=129, dtype='int')[1:] - 1
    tstore = np.linspace(0, s_eval.shape[0]-1, num=s_samp+1, dtype='int')
    dt = s_eval[1] - s_eval[0]
    bc = delta_h(-s_eval, h)
    v1, v2 = v10[ystore, None], v20[ystore, None]
    u_bdy = v10[-1]
    v1temp, v2temp = v10, v20

    # Construct LHS matrix
    lambda_d = dt*eps1*a*b/(2*h**2)
    A = diags([-lambda_d, 1 + 2*lambda_d, -lambda_d], [-1, 0, 1],
              shape=v1temp.shape*2).toarray()

    for i in range(1, s_eval.shape[0]):
        rhsv1 = (v1temp + a*dt/h*(np.append(bc[i-1], v1temp[:-1]) - v1temp)
                 + eps1*a*b*dt/(2*h**2)*(
                    np.append(bc[i-1], v1temp[:-1]) - 2*v1temp
                    + np.append(v1temp[1:], 0)
                 ) + dt/eps2*(-d*v1temp + c*v2temp))
        v1temp = np.linalg.solve(A, rhsv1)
        v2temp += dt/eps2*(d*v1temp - c*v2temp)

        u_bdy = np.append(u_bdy, a*v1temp[-1]
                          - eps1*a*b*(v1temp[-1] - v1temp[-2])
                          - eps1/eps2*c*(a-b)*v2temp[-1])

        if i in tstore:
            v1 = np.append(v1, v1temp[ystore, None], axis=1)
            v2 = np.append(v2, v2temp[ystore, None], axis=1)

    w1 = eps1*a*b*(np.append(bc[None, tstore], v1[:-1], axis=0) - v1)

    u = a*v1 + w1
    v = b*v1 - w1
    f = a*v2  # w2 is 0
    vf = b*v2  # w2 is 0

    return ystore, tstore, u_bdy, u, v, f, vf


if __name__ == '__main__':
    # Define numerical parameters for solution algorithm
    N = 128
    h = 1./N
    s_max = 50
    s_eval = np.linspace(0, s_max, num=s_max*N + 1)

    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N))
    scheme = None

    # Define model parameters
    eps1 = 0.1
    eps2 = 1
    a, c = 0.2, 1
    b, d = 1 - a, 1 - c

    reduced = solve_reduced(s_eval, p0, h, eps1, eps2, a, b, c, d)[2]
    full_tmp = solve_pde(s_eval, p0, h, eps1, eps2, a, b, c, d, scheme)[2:4]
    full = full_tmp[0] + full_tmp[1]

    plt.plot(s_eval, full, label='Full')
    plt.plot(s_eval, reduced, label='Reduced')
    plt.legend()
    plt.show()
