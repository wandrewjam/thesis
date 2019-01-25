"""
File containing the 4 different rolling algorithms.

The four algorithms are named as follows:
1. PDE_eulerian: function to solve the full Eulerian PDE model
2. PDE_bins: function to solve the PDE model with theta-bins
3. stochastic_fixed: function to run a stochastic simulation with a
                     fixed time step
4. stochastic_variable: function to run a stochastic simulation with a
                        variable time step
"""

# I want to compartmentalize pde_eulerian further. I probably need to
# make more assumptions about variables and parameters. E.g. v_f and
# om_f are explicitly given as vectors to the method

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from default_values import *
from utils import nd_force, nd_torque, length


def _handle_velocities(v_f, om_f):
    """
    Converts a scalar v_f and om_f to an array of floats

    Parameters
    ----------
    v_f : int, float, or ndarray
        Applied fluid translational velocity
    om_f : int, float, or ndarray
        Applied fluid rotational velocity

    Returns
    -------
    v_f array : ndarray
        Array of v_f values for each time step in the simulation
    om_f array : ndarray
        Array of om_f values for each time step in the simulation
    """

    assert type(v_f) is float or int or np.ndarray
    assert type(om_f) is float or int or np.ndarray

    if type(v_f) is float or int:
        v_f = (v_f*np.ones(shape=time_steps+1)).astype(dtype=float)
    if type(om_f) is float or int:
        om_f = (om_f*np.ones(shape=time_steps+1)).astype(dtype=float)

    return v_f, om_f


def _cfl_check(v_f, om_f, dt, h, nu, scheme):
    """ Checks the CFL condition """

    if scheme == 'up':
        z_check, om_check = np.max(v_f)*dt/h, np.max(om_f)*dt/nu
    elif scheme == 'bw':
        z_check, om_check = np.max(v_f)*dt/(2*h), np.max(om_f)*dt/(2*nu)
    else:
        raise Exception('the parameter \'scheme\' is not valid')

    if z_check > 1 and om_check > 1:
        raise ValueError('the CFL conditions for both theta and z are not '
                         'satisfied')
    elif z_check > 1:
        raise ValueError('the CFL condition for z is not satisfied')
    elif om_check > 1:
        raise ValueError('the CFL condition for theta is not satisfied')


def _generate_coordinate_arrays(M, N, time_steps, L, T, d):
    """ Generates numerical meshes needed by deterministic algorithms"""

    z_mesh = np.linspace(-L, L, 2*M+1)
    th_mesh = np.linspace(-np.pi/2, np.pi/2, N+1)

    l_mesh = length(z_mesh[:, None], th_mesh[None, :], d)
    t_mesh = np.linspace(0, T, time_steps+1)
    h = z_mesh[1] - z_mesh[0]
    nu = th_mesh[1] - th_mesh[0]
    dt = t_mesh[1] - t_mesh[0]

    return z_mesh, th_mesh, l_mesh, t_mesh, h, nu, dt


def _initialize_unknowns(v_f, om_f, m0, time_steps, save_bond_history):
    """ Initializes arrays of zeros for v, om, and m """
    v, om = np.zeros(shape=v_f.shape), np.zeros(shape=om_f.shape)
    v[0], om[0] = v_f[0], om_f[0]

    if save_bond_history:
        bond_mesh = np.zeros(shape=m0.shape+(time_steps,))
        bond_mesh[:, :, 0] = m0
    else:
        bond_mesh = np.copy(m0)

    return bond_mesh, v, om


def _up(bond_mesh, vel, dx, dt, axis):
    if axis == 0:
        return vel*dt/dx*(bond_mesh[1:, :-1] - bond_mesh[:-1, :-1])
    elif axis == 1:
        return vel*dt/dx*(bond_mesh[:-1, 1:] - bond_mesh[:-1, :-1])


def _bw(bond_mesh, vel, dx, dt, axis):
    if axis == 0:
        return vel*dt/(2*dx)*(-3*bond_mesh[:-2, :-1] + 4*bond_mesh[1:-1, :-1]
                              - bond_mesh[2:, :-1]) + (vel*dt/dx)**2/2\
               * (bond_mesh[:-2, :-1] - 2*bond_mesh[1:-1, :-1]
                  + bond_mesh[2:, :-1])
    elif axis == 1:
        return vel*dt/(2*dx)*(-3*bond_mesh[:-1, :-2] + 4*bond_mesh[:-1, 1:-1]
                              - bond_mesh[:-1, 2:]) + (vel*dt/dx)**2/2\
               * (bond_mesh[:-1, :-2] - 2*bond_mesh[:-1, 1:-1]
                  + bond_mesh[:-1, 2:])


def _form(bond_mesh, form_rate, h, dt, sat):
    sat_term = (1 - sat*np.tile(np.trapz(bond_mesh[:, :-1], dx=h, axis=0),
                                reps=(bond_mesh.shape[0]-1, 1)))
    return dt*form_rate*sat_term


def _count_bonds(bond_mesh, z_mesh, th_mesh, scheme):
    """ Finds the total bond number in a given bond distribution """
    if scheme == 'up':
        return np.trapz(np.trapz(bond_mesh, z_mesh, axis=0), th_mesh, axis=0)
    elif scheme == 'bw':
        return simps(simps(bond_mesh, z_mesh, axis=0), th_mesh, axis=0)
    else:
        raise Exception('parameter \'scheme\' is invalid')


def _eulerian_step(bond_mesh, v, om, h, nu, dt, form_rate, break_rate, sat, scheme):
    new_bonds = np.copy(bond_mesh)
    if scheme == 'up':
        new_bonds[:-1, :-1] += _up(bond_mesh, v, h, dt, axis=0)
        new_bonds[:-1, :-1] += _up(bond_mesh, om, nu, dt, axis=1)
        new_bonds[:-1, :-1] += _form(bond_mesh, form_rate[:-1, :-1], h, dt,
                                     sat)
        new_bonds[:-1, :-1] /= (1 + dt*break_rate[:-1, :-1])
    elif scheme == 'bw':
        new_bonds[:-2, :-1] += _bw(bond_mesh, v, h, dt, axis=0)
        new_bonds[:-1, :-2] += _bw(bond_mesh, om, nu, dt, axis=1)
        new_bonds[-2, :-1] += np.squeeze(_up(bond_mesh[-2:, :], v, h, dt,
                                             axis=0))
        new_bonds[:-1, -2] += np.squeeze(_up(bond_mesh[:, -2:], om, nu, dt,
                                             axis=1))
        new_bonds[:-1, :-1] += _form(bond_mesh, form_rate[:-1, :-1], h, dt,
                                     sat)
        new_bonds[:-1, :-1] /= (1 + dt*break_rate[:-1, :-1])
    return new_bonds


def _run_eulerian_model(bond_mesh, v, om, z_mesh, th_mesh, t_mesh, v_f,
                        om_f, eta_v, eta_om, form_rate, break_rate, sat, d,
                        scheme):
    """ Runs the Eulerian simulation

    This function actually carries out the eulerian timestepping procedure
    to solve the deterministic rolling model.

    Parameters
    ----------
    bond_mesh : ndarray
        An empty 2D (if not save_bond_history) or 3D (if save_bond_history)
        numpy array for storing the bond distribution for (z, theta) or
        (z, theta, t)
    v : ndarray
        A 1D numpy array for storing the translational velocities. The
        first entry is the translational velocity initial condition
    om : ndarray
        A 1D numpy array for storing the rotational velocities. The first
        entry is the rotational velocity initial condition.
    z_mesh : ndarray
        A 1D numpy array containing the numerical mesh in the z dimension
    th_mesh : ndarray
        A 1D numpy array containing the numerical mesh in the theta
        dimension
    ...
    """

    h = z_mesh[1] - z_mesh[0]
    nu = th_mesh[1] - th_mesh[0]
    dt = t_mesh[1] - t_mesh[0]

    if bond_mesh.ndim == 2:  # Equivalent to 'not save_bond_history'
        bond_counts = np.zeros(shape=t_mesh.shape)
        for i in range(t_mesh.shape[0]-1):
            bond_mesh = _eulerian_step(bond_mesh, v[i], om[i], h, nu, dt,
                                       form_rate, break_rate, sat, scheme)
            bond_counts[i+1] = _count_bonds(bond_mesh, z_mesh, th_mesh, scheme)

            current_force = nd_force(bond_mesh, z_mesh[:, None],
                                     th_mesh[None, :])
            current_torque = nd_torque(bond_mesh, z_mesh[:, None],
                                       th_mesh[None, :], d)

            v[i+1] = v_f[i] + current_force/eta_v
            om[i+1] = om_f[i] + current_torque/eta_om
        return bond_counts, v, om
    elif bond_mesh.ndim == 3:
        for i in range(t_mesh.shape[0]):
            bond_mesh = _eulerian_step(bond_mesh, v[i], om[i], h, nu, dt,
                                       form_rate, break_rate, sat, scheme)

            current_force = nd_force(bond_mesh, z_mesh, th_mesh)
            current_torque = nd_torque(bond_mesh, z_mesh, th_mesh, d)

            v[i+1] = v_f[i] + current_force/eta_v
            om[i+1] = om_f[i] + current_torque/eta_om
        return bond_mesh, v, om
    else:
        raise Exception('bond_mesh has an invalid number of dimensions')


def pde_eulerian(M, N, time_steps, m0, scheme='up', **kwargs):
    """ Solves the full eulerian PDE model """

    # Define the problem parameters
    (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history) = set_parameters(**kwargs)

    v_f, om_f = _handle_velocities(v_f, om_f)

    # Define coordinate meshes and mesh widths
    (z_mesh, th_mesh, l_mesh, t_mesh, h, nu, dt) = (
        _generate_coordinate_arrays(M, N, time_steps, L, T, d)
    )

    # Check for the CFL condition
    _cfl_check(v_f, om_f, dt, h, nu, scheme)

    bond_mesh, v, om = _initialize_unknowns(v_f, om_f, m0, time_steps,
                                            save_bond_history)

    # Define distance/length mesh and formation/breaking rate meshes
    form_rate = on*kappa*np.exp(-eta/2*l_mesh**2)
    break_rate = off*np.exp(delta*l_mesh)

    bond_mesh, v, om = _run_eulerian_model(
        bond_mesh, v, om, z_mesh, th_mesh, t_mesh, v_f, om_f, xi_v, xi_om,
        form_rate, break_rate, sat, d, scheme
    )

    return bond_mesh, v, om, t_mesh


if __name__ == '__main__':
    M, N = 32, 32
    time_steps = 2000
    m0 = np.zeros(shape=(2*M+1, N+1))
    model_outputs = []

    for scheme in ['up', 'bw']:
        model_outputs.append(pde_eulerian(M, N, time_steps, m0, scheme=scheme))

    fig, ax = plt.subplots(nrows=3, sharex='all', figsize=(6, 8))

    for i in range(2):
        bond_mesh, v, om, t_mesh = model_outputs[i]
        ax[0].plot(t_mesh, v)
        ax[1].plot(t_mesh, om)
        ax[2].plot(t_mesh, bond_mesh)
    ax[2].set_xlabel('ND time ($s$)')
    ax[0].set_ylabel('ND translation velocity ($v$)')
    ax[1].set_ylabel('ND rotation rate ($\\omega$)')
    ax[2].set_ylabel('ND bond quantity ($\\iint m$)')
    plt.tight_layout()
    plt.show()
