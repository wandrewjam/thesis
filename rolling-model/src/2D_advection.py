import numpy as np
import matplotlib.pyplot as plt

time_steps = 200
M = 100
N = 100
h1, h2 = 1/M, 1/N
dt = 1/time_steps
x, y = np.meshgrid(np.linspace(0, 1, M+1), np.linspace(0, 1, N+1))

u = np.zeros(shape=(M+1, N+1, time_steps+1))
u[:-1, :-1, 0] = np.exp(-((x[1:, 1:] - 0.5)**2 + (y[1:, 1:] - 0.5)**2))

for i in range(time_steps):
    u[:-1, :-1, i+1] = u[:-1, :-1, i] + dt/h1*(u[1:, :-1, i] - u[:-1, :-1, i]) + dt/h2*(u[:-1, 1:, i] - u[:-1, :-1, i])

for i in range(11):
    plt.pcolormesh(x, y, u[:, :, 10*i])
    plt.colorbar()
    plt.show()
