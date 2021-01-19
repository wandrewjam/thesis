import numpy as np
from timeit import default_timer as timer


def main():
    r = np.random.rand(10**8) * 1000

    print('Done generating random numbers')

    start = timer()
    r_sort = np.sort(r)
    end = timer()

    print('Sorting: {}'.format(end - start))
    start = timer()
    s = np.sqrt(r_sort**2 + .01)
    end = timer()

    print('Square root: {}'.format(end - start))

    p = np.linspace(0, 1000, num=10**3)
    sp = np.sqrt(p**2 + .01)

    start = timer()
    rp = np.interp(r_sort, p, sp)
    end = timer()

    print('Interp: {}'.format(end - start))

    print('Error: {}'.format(np.amax(np.abs(s - rp))))


if __name__ == '__main__':
    main()
