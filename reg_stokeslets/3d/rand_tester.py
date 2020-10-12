import numpy as np
from timeit import default_timer as timer


if __name__ == '__main__':
    N = 10 ** 4
    a, b = np.random.rand(N, N), np.random.rand(N, N)

    start = timer()
    a = (((a + b) / b) ** 2) + ((2 * a - 3 * b) / np.pi) ** 3
    end = timer()
    print(end - start)
