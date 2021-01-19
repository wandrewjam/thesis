import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


def pressure(x, y):
    xdiff = x - xk
    ydiff = y - yk
    r2 = xdiff**2 + ydiff**2
    re = np.sqrt(r2 + eps**2)
    p = 0
    for k in range(xk.shape[0]):
        p += (xdiff[k]*fx[k] + ydiff[k]*fy[k])*((r2[k] + 2*eps**2 + eps*re[k]) /
                                                ((re[k]+eps)*re[k]**3))

    return p/(2*np.pi)


def velocity(x, y):
    xdiff = x - xk
    ydiff = y - yk
    r2 = xdiff**2 + ydiff**2
    re = np.sqrt(r2 + eps**2)
    u, v = 0, 0
    for k in range(xk.shape[0]):
        u += -fx[k]/(4*np.pi*mu)*(np.log(re[k] + eps) - eps*(re[k] + 2*eps) /
                                  ((re[k] + eps)*re[k])) + 1/(4*np.pi*mu) *\
             (re[k] + 2*eps)/((re[k] + eps)**2*re[k])*(xdiff[k]**2 * fx[k] + xdiff[k]*ydiff[k] * fy[k])
        v += -fy[k]/(4*np.pi*mu)*(np.log(re[k] + eps) - eps*(re[k] + 2*eps) /
                                  ((re[k] + eps)*re[k])) + 1/(4*np.pi*mu) *\
            (re[k] + 2*eps)/((re[k] + eps)**2*re[k])*(xdiff[k]*ydiff[k] * fx[k] + ydiff[k]**2 * fy[k])
    return u, v


def find_forces(u, v, x, y):
    n = u.shape[0]
    M = np.zeros(shape=(2*n, 2*n))

    for i in range(n):
        for j in range(n):
            xdiff = x[i] - x[j]
            ydiff = y[i] - y[j]

            r2 = xdiff**2 + ydiff**2
            re = np.sqrt(r2 + eps**2)
            a = np.log(re + eps) - eps*(re + 2*eps)/((re + eps)*re)
            b = (re+2*eps)/((re + eps)**2*re)

            tempM = -a*np.eye(2)
            tempM[0, 0] += b*xdiff**2
            tempM[1, 0] += b*xdiff*ydiff
            tempM[0, 1] += b*xdiff*ydiff
            tempM[1, 1] += b*ydiff**2
            M[2*i:2*(i+1), 2*j:2*(j+1)] = tempM/(4*np.pi*mu)

    u_vec = np.zeros(shape=2*n)
    u_vec[::2] = u
    u_vec[1::2] = v

    f_vec = lstsq(M, u_vec)[0]

    return f_vec[::2], f_vec[1::2]


if __name__ == '__main__':
    import sys
    
    N = 200
    u = -np.ones(shape=N)
    v = np.zeros(shape=N)
    mu = 1

    if sys.argv[1] == '--random':
        eps = .03
        xk = np.random.uniform(low=-1, high=1, size=N)
        yk = np.random.uniform(low=-1, high=1, size=N)
    elif sys.argv[1] == '--cylinder':
        xk = 0.25*np.cos(2*np.pi/N*np.arange(N))
        yk = 0.25*np.sin(2*np.pi/N*np.arange(N))
        eps = 2*np.pi*.25/N*.25

    fx, fy = find_forces(u, v, xk, yk)

    x, y = np.meshgrid(np.linspace(-.5, .5), np.linspace(-.5, .5))
    p = np.zeros(shape=x.shape)
    u = np.zeros(shape=x.shape)
    v = np.zeros(shape=x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            p[i, j] = pressure(x[i, j], y[i, j])
            temp_u = velocity(x[i, j], y[i, j])
            u[i, j], v[i, j] = temp_u

    plt.quiver(x, y, u+1, v, np.sqrt((u+1)**2 + v**2))
    plt.axis([-.5, .5, -.5, .5])
    plt.colorbar()
    # plt.contour(x, y, u+1)
    plt.plot(xk, yk, 'ko')
    # plt.show()
    #
    # plt.contour(x, y, v)
    # plt.plot(xk, yk, 'ko')
    plt.show()
