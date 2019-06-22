import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(0, 2, num=500)[1:]

eps, a, b = .01, .2, .8

p = np.sqrt(v/(4*np.pi*eps*a*b))*(a+v)/2*np.exp(-(v - a)**2/(4*eps*a*b*v))

plt.plot(v, p)
plt.show()
