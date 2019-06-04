import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import bmat, coo_matrix, diags, eye
from scipy.sparse.linalg import spsolve


def solve_system(N, h, Dc, Ds, kf, gam):
    """ Solves the scheme for the specified parameter values """
    submatrices = [[None] * N for _ in range(N)]

    submatrices[0][0] = -eye(N, format='coo')
    submatrices[0][1] = diags(np.ones(N - 1), offsets=-1, shape=(N, N - 1),
                              format='coo')
    # Construct the rest of the block matrix
    for i in range(1, N):
        diagonal = np.ones(N - i)
        diagonal[-1] = 0
        # Construct the block sub-diagonal
        submatrices[i][i - 1] = ((Ds/h**2 + kf/(2*h*gam) * (z[i] - lam))
                                 * diags(diagonal, offsets=1,
                                         shape=(N - i, N - i + 1),
                                         format='coo'))
        # Construct the block diagonal
        part1 = diags(diagonal, offsets=0, shape=(N - i, N - i))
        if i < N - 1:
            part2 = coo_matrix(([1, -1], ([N - i - 1, N - i - 1],
                                          [N - i - 2, N - i - 1])),
                               shape=(N - i, N - i))
        elif i == N - 1:
            part2 = coo_matrix(([-1], ([0], [0])), shape=(1, 1))
        else:
            raise ValueError
        part3 = diags([diagonal[1:], diagonal[:-1]], offsets=[-1, 1],
                      shape=(N - i, N - i), format='coo')
        submatrices[i][i] = (-2 * (Dc + Ds)/h**2 * part1 + part2
                             + Dc/h**2 * part3)
        # Construct the block super-diagonal
        if i < N - 1:
            submatrices[i][i + 1] = ((Ds/h**2 - kf/(2*h*gam) * (z[i] - lam))
                                     * diags(diagonal[1:], offsets=-1,
                                             shape=(N - i, N - i - 1),
                                             format='coo'))
    # Construct a single matrix from the array of sub-matrices
    A = bmat(submatrices, format='csr')
    # Construct the rhs vector
    rhs = np.zeros(N)
    temp = -np.ones(N)
    temp[-1] = 0
    for i in range(1, N):
        rhs = np.append(rhs, temp[i:])

    # Solve the system
    sol = spsolve(A, rhs)
    return sol


def generate_triangulation_arrays(x, z):
    """ Generates point and triangle arrays for domain triangulation """
    assert x.shape == z.shape

    N = x.shape[0] - 1

    xtri = []
    for i in range(N + 1):
        xtri.append(x[i:])
    xtri = np.hstack(xtri)
    ztri = np.repeat(z, np.arange(1, N + 2)[::-1])
    npoints = xtri.shape[0]

    assert npoints == (N+1)*(N+2)/2

    trirows = []
    for i in range(N + 1, 1, -1):
        ti = i * (i + 1)/2
        offset = npoints - ti
        for j in range(0, i - 1):
            trirows.append(np.array([offset + j, offset + j + 1, offset + j + i]))
        if i > 2:
            for j in range(1, i - 1):
                trirows.append(np.array([offset + j, offset + j + i,
                                         offset + j + i - 1]))
    tri = np.vstack(trirows)
    return xtri, ztri, tri


def inject_boundaries(sol):
    """ Inject 0 boundary along x=z into the solution vector """
    obj = []
    tn = N*(N + 1)/2
    assert tn == sol.shape[0]

    for i in range(N, 0, -1):
        obj.append(tn - i*(i + 1)/2)
    plot_sol = np.insert(sol, obj=np.array(obj), values=0)
    plot_sol = np.append(plot_sol, 0)
    return plot_sol


if __name__ == '__main__':
    import sys
    
    # Define parameters
    Ds, Dc = 1, 1
    kf, gam = 1, 1
    lam = 0
    N = 100
    L = 5

    z = np.linspace(0, L, num=N+1)
    x = np.linspace(0, L, num=N+1)
    h = x[1] - x[0]

    sol = solve_system(N, h, Dc, Ds, kf, gam)

    xtri, ztri, tri = generate_triangulation_arrays(x, z)

    plot_sol = inject_boundaries(sol)

    plt.rcParams['figure.figsize'] = (8, 6)
    triangulation = Triangulation(xtri, ztri, tri)
    # plt.triplot(triangulation, 'k-', linewidth=0.1)
    cs = plt.tricontour(triangulation, plot_sol, colors='k')
    plt.clabel(cs)
    plt.tripcolor(triangulation, plot_sol, shading='gouraud')
    plt.xlabel('$x$')
    plt.ylabel('$z$')
    plt.colorbar()
    plt.savefig('binding-times.png', dpi=300)
    plt.show()

    # Generate a sample mesh
    Nsamp = 5
    xsamp, zsamp = (np.linspace(0, L, num=Nsamp+1),
                    np.linspace(0, L, num=Nsamp+1))

    xtrisamp, ztrisamp, trisamp = generate_triangulation_arrays(xsamp, zsamp)

    fig, ax = plt.subplots()
    ax.triplot(xtrisamp, ztrisamp, trisamp, 'k-o')
    ax.set_aspect('equal')
    plt.savefig('sample-mesh.png', dpi=300)
    plt.show()
