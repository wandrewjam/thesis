import numpy as np
from timeit import default_timer as timer


if __name__ == '__main__':
    size = 10**4

    a = np.random.rand(size, size)
    ind_u = np.triu_indices(size, k=1)
    a2 = np.triu(a)
    am = np.ma.array(a)
    am[ind_u[::-1]] = np.ma.masked

    r = np.linspace(0, 1, num=100)
    sq = np.sqrt(r**2 + .01)

    # start = timer()
    # res1 = np.sqrt(a**2 + .01)
    # end = timer()
    # print(end - start)

    start = timer()
    res2 = np.sqrt(a2**2 + .01)
    res2[ind_u[::-1]] = res2[ind_u]
    end = timer()
    print(end - start)

    assert np.all(res2 == res2.T)

    start = timer()
    res5 = np.sqrt(am**2 + .01)
    a3 = np.zeros(shape=(size, size))
    a3[ind_u] = am[ind_u]
    a3[ind_u[::-1]] = am[ind_u]
    end = timer()
    print(end - start)

    start = timer()
    res3 = np.interp(a, r, sq)
    end = timer()
    print(end - start)

    start = timer()
    res4 = np.interp(a2, r, sq)
    res4[ind_u[::-1]] = res4[ind_u]
    end = timer()
    print(end - start)
