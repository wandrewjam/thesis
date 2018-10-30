import numpy as np
import matplotlib.pyplot as plt


def cart_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    x[np.where(x == 0)] = np.finfo(float).tiny
    theta = np.arctan(y/x)
    return r, theta


def polar_to_cart(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def ur(r, theta):
    return U*(1 - a**2/r**2)*np.cos(theta)


def uth(r, theta):
    return -U*(1 + a**2/r**2)*np.sin(theta) + gamma/(2*np.pi*r)


U, a, gamma = 1, 1, 0

r, theta = np.meshgrid(np.linspace(1, 5, 1000)[1:],
                       np.linspace(0, 2*np.pi, 1000)[:-1])
u
