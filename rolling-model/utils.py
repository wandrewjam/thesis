# Base functions for rolling models

import numpy as np
from scipy.sparse import block_diag, diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapz


def length(z, th, d=0):
    """
    Computes the distance between z and theta

    Parameters
    ----------
    z, th : array_like
        Coordinate arrays specifying points on the wall and the platelet,
        respectively.
    d : float
        The minimum separation distance between the platelet and wall.

    Returns
    -------
    output : array_like
        Returns the distance(s) between z and theta. If z and theta are
        both scalars, then a scalar is returned. Otherwise an array is
        returned.

    Raises
    ------
    ValueError
        If z and th cannot be broadcast to a single shape.
    """
    return np.sqrt((1 - np.cos(th) + d)**2 + (np.sin(th) - z)**2)


def construct_system(M, N, eta, z_vec, th_vec, delta, nu, kap, d_prime=0,
                     saturation=True):
    # Construct A matrix (upwind in theta)
    diagonals = [-np.ones((2*M+1)*(N+1)), np.ones((2*M+1)*N)]
    offsets = [0, 2*M+1]
    A = diags(diagonals, offsets)

    # Construct B matrix (upwind in z)
    subdiag = np.ones((2*M+1)*(N+1) - 1)
    subdiag[(2*M)::(2*M+1)] = 0
    diagonals = [-np.ones((2*M+1)*(N+1)), subdiag]
    offsets = [0, 1]
    B = diags(diagonals, offsets)

    # Construct C matrix (bond formation term)
    if saturation:
        left_matrix = diags(np.exp(-eta*length(z_vec, th_vec, d_prime)/2))
        right_matrix = block_diag((-np.ones((2*M+1, 2*M+1)),)*(N+1))
        C = np.dot(left_matrix, right_matrix)
    else:
        C = diags(np.exp(-eta*length(z_vec, th_vec, d_prime)/2))

    # Construct D matrix (bond breaking term)
    D = diags(-np.exp(delta*length(z_vec, th_vec, d_prime)))

    # Construct RHS
    R = nu*kap*np.exp(-eta*length(z_vec, th_vec, d_prime)**2/2)

    return A, B, C, D, R


def nd_torque(m_mesh, z_mesh, th_mesh, d_prime=0):
    tau = trapz(trapz(((1 - np.cos(th_mesh) + d_prime)*np.sin(th_mesh) +
                       (np.sin(th_mesh) - z_mesh)*np.cos(th_mesh))*m_mesh, x=z_mesh,
                      axis=0), x=th_mesh[0,])
    return tau


def nd_force(m_mesh, z_mesh, th_mesh):
    # Function to calculate net force given a distribution of bonds, m
    f_prime = trapz(trapz((z_mesh - np.sin(th_mesh))*m_mesh, x=z_mesh, axis=0),
                    x=th_mesh[0,])
    return f_prime


def find_torque_roll(A, B, C, D, R, om, v, lam, nu, h, kap, M, z_mesh, th_mesh,
                     d_prime=0):
    # Calculate the bond density function
    m = spsolve(om*A + v*lam*B + h*nu*kap*C + nu*D, -R)
    m_mesh = m.reshape(2*M+1, -1, order='F')
    return nd_torque(m_mesh, z_mesh, th_mesh, d_prime), m_mesh


def find_force_roll(A, B, C, D, R, om, v, lam, nu, h, kap, M, z_mesh, th_mesh,
                    d_prime=0):
    # Calculate the bond density function
    m = spsolve(om*A + v*lam*B + h*nu*kap*C + nu*D, -R)
    m_mesh = m.reshape(2*M+1, -1, order='F')
    return nd_force(m_mesh, z_mesh, th_mesh), m_mesh


def find_forces(A, B, C, D, R, om, v, lam, nu, h, kap, M, z_mesh, th_mesh,
                d_prime=0):
    # Calculate the bond density function
    m = spsolve(om*A + v*lam*B + h*nu*kap*C + nu*D, -R)
    m_mesh = m.reshape(2*M+1, -1, order='F')
    return nd_torque(m_mesh, z_mesh, th_mesh, d_prime), \
        nd_force(m_mesh, z_mesh, th_mesh), m_mesh