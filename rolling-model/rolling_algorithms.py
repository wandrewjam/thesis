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
import matplotlib.pyplot as plt

from default_values import set_parameters
from constructA import nd_force, nd_torque, length


def _handle_velocities(v_f, om_f):
    """
    This function converts a scalar v_f and om_f to an array of
    floats, if necessary
    """

    if type(v_f) is float or int:
        v_f = (v_f*np.ones(shape=time_steps+1)).astype(dtype=float)
    if type(om_f) is float or int:
        om_f = (om_f*np.ones(shape=time_steps+1)).astype(dtype=float)

    return v_f, om_f


def _cfl_check(v_f, om_f, dt, h, nu, scheme):
    """ This function checks the CFL condition """

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
    z_mesh = np.linspace(-L, L, 2*M+1)
    th_mesh = np.linspace(-np.pi/2, np.pi/2, N+1)
    l_mesh = length(z_mesh, th_mesh[None, :], d)
    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1] - z_mesh[0]
    nu = th_mesh[1] - th_mesh[0]
    dt = t[1] - t[0]

    return z_mesh, th_mesh, l_mesh, t, h, nu, dt


def _initialize_unknowns(v_f, om_f, m0, time_steps, save_bond_history):
    v, om = np.zeros(shape=v_f.shape), np.zeros(shape=om_f)
    v[0], om[0] = v_f[0], om_f[0]

    if save_bond_history:
        bond_mesh = np.zeros(shape=m0.shape+(time_steps,))
        bond_mesh[:, :, 0] = m0
    else:
        bond_mesh = np.copy(m0)

    return v, om, bond_mesh


def _upwind(v, om, bond_mesh, v_f, om_f, form_rate, break_rate,
            save_bond_history):
    """ This function carries out the upwind scheme """

    time_steps = v.shape[0]

    if save_bond_history:
        for i in range(time_steps + 1):
            bond_mesh[:, :, i+1] = _upwind_step(bond_mesh[:, :, i])

    else:
        for i in range(time_steps + 1):
            bond_mesh = _upwind_step(bond_mesh)

    return v, om, bond_mesh, forces, torques


def pde_eulerian(M, N, time_steps, m0, **kwargs):
    """This is a function to solve the full eulerian PDE model."""

    # Define the problem parameters.
    (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history, scheme) = set_parameters(**kwargs)

    v_f, om_f = _handle_velocities(v_f, om_f)

    # Define coordinate meshes and mesh widths.
    (z_mesh, th_mesh, l_mesh, t, h, nu, dt) = (
        _generate_coordinate_arrays(M, N, time_steps, L, T, d)
    )

    # Check for the CFL condition
    _cfl_check(v_f, om_f, dt, h, nu)

    v, om, bond_mesh = _initialize_unknowns(v_f, om_f, m0, time_steps,
                                            save_bond_history)

    # Define distance/length mesh and formation/breaking rate meshes.
    form_rate = on*kappa*np.exp(-eta/2*l_mesh[:-1, :-1]**2)
    break_rate = off*np.exp(delta*l_mesh[:-1, :-1])

    def time_step(v_f, om_f, v, om, bond_mesh):
        """
        Iterate the PDE using 1st order upwind, and velocity values
        from the previous time step.
        """
        bond_mesh[:-1, :-1] = (
            (bond_mesh[:-1, :-1]
             + om*dt/nu*(bond_mesh[:-1, 1:] - bond_mesh[:-1, :-1])
             + v*dt/h*(bond_mesh[1:, :-1] - bond_mesh[:-1, :-1])
             + dt*form_rate*(1 - sat*np.tile(np.trapz(bond_mesh[:, :-1],
                                                      z_mesh, axis=0),
                                             reps=(2*M, 1))))
            / (1 + dt*break_rate)
        )

        if xi_v == np.inf:
            # Set linear velocity to fluid velocity
            new_v = v_f
        else:
            # Calculate bond forces and balance with fluid forces
            f_prime = nd_force(bond_mesh, z_mesh, th_mesh)
            new_v = v_f + f_prime/xi_v

        if xi_om == np.inf:
            # Set angular velocity to fluid angular velocity
            new_om = om_f
        else:
            # Calculate bond torques and balance with fluid torques
            tau = nd_torque(bond_mesh, z_mesh, th_mesh, d)
            new_om = om_f + tau/xi_om

        return bond_mesh, new_v, new_om

    # if save_bond_history:
    #     m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))
    #     m_mesh[:, :, 0] = m0
    #
    #     for i in range(time_steps):
    #         m_mesh[:, :, i+1], v[i+1], om[i+1] = (
    #             time_step(v_f[i], om_f[i], v[i], om[i],
    #                       bond_mesh=m_mesh[:, :, i])
    #         )
    #
    # else:
    #     m_mesh = m0
    #
    #     for i in range(time_steps):
    #         m_mesh, v[i+1], om[i+1] = (
    #             time_step(v_f[i], om_f[i], v[i], om[i], bond_mesh=m_mesh)
    #         )

    if scheme == 'up':
        v, om, bond_mesh, forces, torques = (
            _upwind(v, om, bond_mesh, v_f, om_f, form_rate, break_rate)
        )
    elif scheme == 'bw':
        v, om, bond_mesh, forces, torques = _beam_warming(v, om, bond_mesh,
                                                          form_rate, break_rate)

    return v, om, bond_mesh, t


if __name__ == '__main__':
    M, N = 128, 128
    time_steps = 1000
    m0 = np.zeros(shape=(2*M+1, N+1))
    v, om, m_mesh, t = pde_eulerian(M, N, time_steps, m0)

    plt.plot(t, v)
    plt.show()
