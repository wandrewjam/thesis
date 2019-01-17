import numpy as np
from utils import nd_force
from scipy.stats import truncnorm

# A script to test the force functions


def bond_dist(z_mesh, th_mesh):
    return np.exp(-(z_mesh + np.sqrt(2)/2)**2 - (th_mesh + np.pi/4)**2)


L = 2.5
M, N = 2**7
eta = 0.1
z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1), np.linspace(-np.pi/2, np.pi/2, N+1), indexing='ij')

m_mesh = bond_dist(z_mesh, th_mesh)
force1 = nd_force(m_mesh, z_mesh, th_mesh)

z_list = truncnorm.rvs(a=(-L+np.sqrt(2)/2)*np.sqrt(1/eta), b=(L + np.sqrt(2)/2)*np.sqrt(1/eta), )
