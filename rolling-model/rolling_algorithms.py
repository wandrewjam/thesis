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


def pde_eulerian(M, N, time_steps, m0, **kwargs):
    """This is a function to solve the full eulerian PDE model."""

    # Define the problem parameters.
    (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history) = set_parameters(**kwargs)

    if type(v_f) is float or int:
        v_f = v_f*np.ones(shape=time_steps+1)
    if type(om_f) is float or int:
        om_f = om_f*np.ones(shape=time_steps+1)

    # Define coordinate meshes and mesh widths.
    z_mesh, th_mesh = np.meshgrid(np.linspace(-L, L, 2*M+1),
                                  np.linspace(-np.pi/2, np.pi/2, N+1),
                                  indexing='ij')
    t = np.linspace(0, T, time_steps+1)
    h = z_mesh[1, 0] - z_mesh[0, 0]
    nu = th_mesh[0, 1] - th_mesh[0, 0]
    dt = t[1]-t[0]

    # Define distance/length mesh and formation/breaking rate meshes.
    l_matrix = length(z_mesh, th_mesh, d)
    form_rate = on*kappa*np.exp(-eta/2*l_matrix[:-1, :-1]**2)
    break_rate = off*np.exp(delta*l_matrix[:-1, :-1])

    # Check for the CFL condition
    if np.max(om_f)*dt/nu > 1 and np.max(v_f)*dt/h > 1:
        raise ValueError('the CFL conditions for theta and z are not '
                         'satisfied')
    elif np.max(v_f)*dt/h > 1:
        raise ValueError('the CFL condition for z is not satisfied!')
    elif np.max(om_f)*dt/nu > 1:
        raise ValueError('the CFL condition for theta is not satisfied!')

    # Initialize velocity vectors
    v = np.zeros(shape=time_steps+1)
    v[0] = v_f[0]

    om = np.zeros(shape=time_steps+1)
    om[0] = om_f[0]

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
                                                      z_mesh[:, 0], axis=0),
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

    if save_bond_history:
        m_mesh = np.zeros(shape=(2*M+1, N+1, time_steps+1))
        m_mesh[:, :, 0] = m0

        for i in range(time_steps):
            m_mesh[:, :, i+1], v[i+1], om[i+1] = (
                time_step(v_f[i], om_f[i], v[i], om[i],
                          bond_mesh=m_mesh[:, :, i])
            )

    else:
        m_mesh = m0

        for i in range(time_steps):
            m_mesh, v[i+1], om[i+1] = (
                time_step(v_f[i], om_f[i], v[i], om[i], bond_mesh=m_mesh)
            )

    return v, om, m_mesh, t


if __name__ == '__main__':
    M, N = 128, 128
    time_steps = 1000
    m0 = np.zeros(shape=(2*M+1, N+1))
    v, om, m_mesh, t = pde_eulerian(M, N, time_steps, m0)

    plt.plot(t, v)
    plt.show()
