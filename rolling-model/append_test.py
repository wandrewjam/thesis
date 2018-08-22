import numpy as np
from timeit import default_timer as timer

nrows = 2**3
rows = np.random.rand(nrows, 2)

start1 = timer()
x = np.zeros(shape=(0, 2))
x = np.append(arr=x, values=rows, axis=0)
end1 = timer()

print(end1 - start1)

start2 = timer()
x = np.zeros(shape=(0, 2))
for i in range(nrows):
    x = np.append(arr=x, values=rows[i:i+1, :], axis=0)
end2 = timer()

print(end2 - start2)
