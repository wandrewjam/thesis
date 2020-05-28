import numpy as np
import multiprocessing as mp
from itertools import product
from timeit import default_timer as timer


def main():
    n = 5 * 10**4
    m = 3
    a = np.random.rand(n, n, 3, 3)
    b = np.random.rand(3, 3)

    ## start = timer()
    ## result1 = np.dot(a, b)
    ## end = timer()

    ## print(result1.shape)
    ## print(end - start)
    
    def helper(arr):
        return np.dot(arr, b)

    pool = mp.Pool(processes=32)

    slice_size = n // 32

    indices = [np.arange(i * slice_size, (i+1) * slice_size) for i in range(31)]
    indices.append(np.arange(31 * slice_size, n))
    
    r = range(n)
    start = timer()
    result2 = [pool.apply_async(helper, args=(a[slice])) for slice in indices]
    end = timer()
    
    print(type(result2))
    print(end - start)
        

if __name__ == '__main__':
    main()
