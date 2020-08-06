import numpy as np


if __name__ == '__main__':
    n = 2**14
    a = np.random.rand(n, n)
    b = np.random.rand(n)

    s = np.linalg.solve(a, b)
    print(s)
