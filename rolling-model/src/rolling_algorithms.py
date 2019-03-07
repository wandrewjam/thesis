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

# Really the way to do this is to use a more object-oriented approach in order
# to easily pass the necessary parameters between functions

import multiprocessing as mp
import numpy as np
from scipy.integrate import simps
from scipy.special import erf
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from default_values import *
from utils import nd_force, nd_torque, length
from timeit import default_timer as timer


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


def _generate_coordinate_arrays(M, N, time_steps, L, T, d, alg):
    """ Generates numerical meshes needed by simulation algorithms"""

    z_mesh = np.linspace(-L, L, 2*M+1)
    if alg == 'det':
        th_mesh = np.linspace(-np.pi/2, np.pi/2, N+1)
    elif alg == 'sto':
        th_mesh = np.linspace(-np.pi, np.pi, 2*N+1)[:-1]
    else:
        raise Exception('parameter \'alg\' is not valid')

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
        bond_mesh = np.zeros(shape=m0.shape+(time_steps+1,))
        bond_mesh[:, :, 0] = m0
    else:
        bond_mesh = np.copy(m0)

    return bond_mesh, v, om


def _initialize_unknown_lists(v_f, om_f, m0, save_bond_history):
    """ Initializes arrays for v, om, and bond_list """

    v, om = np.array([v_f[0]]), np.array([om_f[0]])
    t_mesh = np.array([0])

    if save_bond_history:
        master_list = np.zeros(shape=0)
        master_list = np.append(arr=master_list, values=m0)
    else:
        master_list = m0

    return master_list, v, om, t_mesh


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


def _form(bond_mesh, form_rate, h, dt, sat, scheme):
    if scheme == 'up':
        integrator = np.trapz
    elif scheme == 'bw':
        integrator = simps
    else:
        raise Exception('Parameter \'scheme\' is invalid')
    sat_term = (1 - sat*np.tile(integrator(bond_mesh[:, :-1], dx=h, axis=0),
                                reps=(bond_mesh.shape[0]-1, 1)))
    return dt*form_rate*sat_term


def _interpolate_velocities(v_f, om_f, T):
    """ Returns functions to interpolate fluid velocity data """

    assert v_f.shape == om_f.shape, 'velocity shapes are not correct'

    t_uniform = np.linspace(0, T, num=v_f.shape[0])

    return (
        interp1d(t_uniform, v_f, kind='linear', bounds_error=False,
                 fill_value='extrapolate'),
        interp1d(t_uniform, om_f, kind='linear', bounds_error=False,
                 fill_value='extrapolate')
    )


def _frac(th_mesh, nu):

    assert nu > 0, 'nu is invalid'  # Checks that nu is valid

    def ramp(th):
        return ((np.pi + nu)/2 - np.abs(th))/nu

    th_vec = th_mesh.flatten(order='F')
    fraction = np.piecewise(th_vec,
                            [np.pi/2 + nu/2 <= np.abs(th_vec),
                             np.abs(np.pi/2 - np.abs(th_vec)) <= nu/2,
                             np.abs(th_vec) <= (np.pi - nu)/2],
                            [0, ramp, 1])

    # Makes sure factor is in the correct range
    assert np.min(fraction) >= 0 and np.max(fraction) <= 1, \
        'frac is not in the correct range'

    return fraction.reshape(th_mesh.shape, order='F')


def _find_rates(binned_bonds, bond_lengths, on, off, bond_max, sat, expected_coeffs, delta):
    """ Finds breaking and formation rates for the stochastic model"""

    break_rates = off*np.exp(delta*bond_lengths)
    form_rates = on*expected_coeffs*(bond_max - sat*binned_bonds)
    all_rates = np.append(break_rates, form_rates)

    return all_rates


def _get_next_reaction(all_rates, max_step):
    """ Finds the next reaction to occur in the stochastic simulation """

    sum_rates = np.cumsum(all_rates)
    total_rate = sum_rates[-1]

    r = np.random.rand(2)
    try:
        dt = 1/total_rate * np.log(1/r[0])
    except ZeroDivisionError:
        dt = np.inf

    if dt > max_step:
        return max_step, None
    else:
        j = np.searchsorted(a=sum_rates, v=r[1]*total_rate)
        return dt, j


def _update_bond_positions(bond_list, th_mesh, t_mesh, dt, v, om):
    """ Updates bond positions and time mesh for ssa """

    bond_list[:, 0] += -dt*v  # Doesn't handle the case v not constant
    th_mesh += -dt*om  # Same here
    th_mesh = ((th_mesh + np.pi) % (2*np.pi)) - np.pi  # Should I put
                                            # this in its own function?
    t_mesh = np.append(t_mesh, t_mesh[-1] + dt)

    return bond_list, th_mesh, t_mesh


def _advect_bonds_out(bond_list, th_mesh, nu, dt, om, L, correct_flux):
    """ Breaks bonds that advect out of numerical domain """

    break_indices = np.where(th_mesh[bond_list[:, 1].astype(int)]
                             < -(np.pi + nu)/2)
    break_indices = np.append(break_indices, values=np.where(
        th_mesh[bond_list[:, 1].astype(int)] > np.pi/2))
    break_indices = np.append(break_indices, values=np.where(
        bond_list[:, 0] > L))
    break_indices = np.append(break_indices, values=np.where(
        bond_list[:, 0] < -L))

    if correct_flux:
        last_bin = np.nonzero((th_mesh < -(np.pi - nu)/2)
                              * (th_mesh > -(np.pi + nu)/2))
        fraction = dt*om/(nu*_frac(th_mesh[last_bin] + dt*om, nu))
        fraction = np.maximum(0, fraction)
        assert 0 <= fraction < 1, 'fraction is outside the correct range'

        lbin_rows = np.where(bond_list[:, 1].astype(int) == last_bin)[1]
        r_advect = np.random.binomial(1, fraction, size=lbin_rows.shape)
        break_indices = np.append(break_indices,
                                  values=lbin_rows[np.nonzero(r_advect)])
    return break_indices


def _get_forces(bond_list, th_mesh, bond_max, nu, d):
    """ Calculates the force and torque on a platelet """

    zs = bond_list[:, 0]
    thetas = bond_list[:, 1].astype(int)
    force = nu/bond_max*np.sum(a=zs-np.sin(th_mesh[thetas]))
    torque = nu/bond_max*np.sum(
        a=((1-np.cos(th_mesh[thetas]) + d) * np.sin(th_mesh[thetas])
           + (np.sin(th_mesh[thetas])-zs) * np.cos(th_mesh[thetas]))
    )

    return force, torque


def _count_bonds(bond_mesh, z_mesh, th_mesh, scheme):
    """ Finds the total bond number in a given bond distribution """
    if scheme == 'up':
        return np.trapz(np.trapz(bond_mesh, z_mesh, axis=0), th_mesh, axis=0)
    elif scheme == 'bw':
        return simps(simps(bond_mesh, z_mesh, axis=0), th_mesh, axis=0)
    else:
        raise Exception('parameter \'scheme\' is invalid')


def _get_avg_length(bond_mesh, z_mesh, th_mesh, d, scheme):
    """ Calculate the average length of a bond distribution """

    if scheme == 'up':
        return (
                np.trapz(np.trapz(length(z_mesh[:, None], th_mesh[None, :], d)
                                  * bond_mesh, z_mesh, axis=0), th_mesh, axis=0)
                / _count_bonds(bond_mesh, z_mesh, th_mesh, scheme)
        )
    elif scheme == 'bw':
        return (
                simps(simps(length(z_mesh[:, None], th_mesh[None, :], d)
                            * bond_mesh, z_mesh, axis=0), th_mesh, axis=0)
                / _count_bonds(bond_mesh, z_mesh, th_mesh, scheme)
        )
    else:
        raise Exception('parameter \'scheme\' is invalid')


def _eulerian_step(bond_mesh, v, om, h, nu, dt, form_rate, break_rate, sat,
                   scheme):
    new_bonds = np.copy(bond_mesh)
    if scheme == 'up':
        new_bonds[:-1, :-1] += _up(bond_mesh, v, h, dt, axis=0)
        new_bonds[:-1, :-1] += _up(bond_mesh, om, nu, dt, axis=1)
        new_bonds[:-1, :-1] += _form(bond_mesh, form_rate[:-1, :-1], h, dt,
                                     sat, scheme)
        new_bonds[:-1, :-1] /= (1 + dt*break_rate[:-1, :-1])
    elif scheme == 'bw':
        new_bonds[:-2, :-1] += _bw(bond_mesh, v, h, dt, axis=0)
        new_bonds[:-1, :-2] += _bw(bond_mesh, om, nu, dt, axis=1)
        new_bonds[-2, :-1] += np.squeeze(_up(bond_mesh[-2:, :], v, h, dt,
                                             axis=0))
        new_bonds[:-1, -2] += np.squeeze(_up(bond_mesh[:, -2:], om, nu, dt,
                                             axis=1))
        new_bonds[:-1, :-1] += _form(bond_mesh, form_rate[:-1, :-1], h, dt,
                                     sat, scheme)
        new_bonds[:-1, :-1] /= (1 + dt*break_rate[:-1, :-1])
    return new_bonds


def _update_bond_list(bond_list, th_mesh, j, break_indices, a, b, eta):
    """ Updates the bond list """

    if j is not None:  # First reaction occurs before min_step
        if j < bond_list.shape[0]:
            break_indices = np.append(break_indices, j)
        else:
            index = j - bond_list.shape[0]
            new_bonds = np.zeros(shape=(1, 2))
            new_bonds[0, 0] = truncnorm.rvs(a=a[index], b=b[index],
                                            loc=np.sin(th_mesh[index]),
                                            scale=np.sqrt(1/eta))
            new_bonds[0, 1] = index
            bond_list = np.append(arr=bond_list, values=new_bonds, axis=0)

        bond_list = np.delete(arr=bond_list, obj=break_indices, axis=0)

    return bond_list


def _run_eulerian_model(bond_mesh, v, om, z_mesh, th_mesh, t_mesh, h, nu, dt,
                        v_f, om_f, eta_v, eta_om, form_rate, break_rate, sat,
                        d, scheme):
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

    # avg_lengths = np.zeros(shape=t_mesh.shape)
    # avg_lengths[0] = _get_avg_length(bond_mesh, z_mesh, th_mesh, scheme)

    if bond_mesh.ndim == 2:  # Equivalent to 'not save_bond_history'
        bond_counts = np.zeros(shape=t_mesh.shape)
        bond_counts[0] = _count_bonds(bond_mesh, z_mesh, th_mesh, scheme)
        avg_lengths = np.zeros(shape=t_mesh.shape)
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
            avg_lengths[i+1] = _get_avg_length(bond_mesh, z_mesh, th_mesh, d,
                                               scheme)
    elif bond_mesh.ndim == 3:
        for i in range(t_mesh.shape[0]-1):
            bond_mesh[:, :, i+1] = _eulerian_step(bond_mesh[:, :, i], v[i],
                                                  om[i], h, nu, dt, form_rate,
                                                  break_rate, sat, scheme)

            current_force = nd_force(bond_mesh[:, :, i+1], z_mesh[:, None],
                                     th_mesh[None, :])
            current_torque = nd_torque(bond_mesh[:, :, i+1], z_mesh[:, None],
                                       th_mesh[None, :], d)

            v[i+1] = v_f[i] + current_force/eta_v
            om[i+1] = om_f[i] + current_torque/eta_om
        bond_counts = _count_bonds(bond_mesh, z_mesh, th_mesh, scheme)
    else:
        raise Exception('bond_mesh has an invalid number of dimensions')
    return bond_mesh, bond_counts, v, om, avg_lengths


def _run_stochastic_model(master_list, v, om, t_mesh, th_mesh, L, T, nu,
                          bond_max, d, v_f, om_f, eta_v, eta_om, eta, delta,
                          on, off, sat, correct_flux, max_step_length,
                          coeffs_and_bounds):
    """ Runs the variable time-step stochastic algorithm """

    v_f_interp, om_f_interp = _interpolate_velocities(v_f, om_f, T)

    force_arr, torque_arr = _get_forces(master_list, th_mesh, bond_max, nu, d)
    if master_list.ndim == 2:  # Equivalent to 'not save_bond_history
        bond_counts = np.zeros(shape=0)
        bond_counts = np.append(arr=bond_counts, values=master_list.shape[0])
        avg_lengths = np.zeros(shape=0)
        while t_mesh[-1] < T:
            bond_lengths = length(master_list[:, 0],
                                  th_mesh[master_list[:, 1].astype(int)], d=d)

            binned_bonds = np.bincount(master_list[:, 1].astype(int),
                                       minlength=2*N)
            expected_coeffs, a, b = coeffs_and_bounds(th_mesh)

            all_rates = _find_rates(binned_bonds, bond_lengths, on, off,
                                    bond_max, sat, expected_coeffs, delta)

            assert max_step_length > 0
            max_step = max_step_length / np.maximum(np.amax(v_f), np.abs(v[-1]))
            assert max_step > 0
            dt, j = _get_next_reaction(all_rates, max_step)
            master_list, th_mesh, t_mesh = (
                _update_bond_positions(master_list, th_mesh, t_mesh, dt,
                                       v[-1], om[-1])
            )

            break_indices = _advect_bonds_out(master_list, th_mesh, nu, dt,
                                              om[-1], L, correct_flux)

            master_list = _update_bond_list(master_list, th_mesh, j,
                                            break_indices, a, b, eta)

            bond_counts = np.append(arr=bond_counts,
                                    values=master_list.shape[0])
            force, torque = _get_forces(master_list, th_mesh, bond_max, nu, d)
            force_arr = np.append(force_arr, force)
            torque_arr = np.append(torque_arr, torque)

            v = np.append(arr=v, values=v_f_interp(t_mesh[-1]) + force/eta_v)
            om = np.append(arr=om, values=om_f_interp(t_mesh[-1])
                                          + torque/eta_om)
            try:
                avg_lengths = np.append(arr=avg_lengths,
                                        values=np.mean(bond_lengths))
            except RuntimeWarning:
                pass

    elif master_list.ndim == 1:
        while t_mesh[-1] < T:
            bond_list = np.copy(master_list[-1])
            assert bond_list.ndim == 2

            bond_lengths = length(bond_list[:, 0],
                                  th_mesh[bond_list[:, 1].astype(int)], d=d)

            binned_bonds = np.bincount(bond_list[:, 1].astype(int),
                                       minlength=2*N)
            expected_coeffs, a, b = coeffs_and_bounds(th_mesh)

            all_rates = _find_rates(binned_bonds, bond_lengths, on, off,
                                    bond_max, sat, expected_coeffs)

            assert v[-1] > 0
            max_step = max_step_length / np.abs(v[-1])
            dt, j = _get_next_reaction(all_rates, max_step)
            bond_list, th_mesh, t_mesh = (
                _update_bond_positions(bond_list, th_mesh, t_mesh, dt)
            )

            break_indices = _advect_bonds_out(bond_list, th_mesh, nu, dt, om,
                                              L, correct_flux)

            bond_list = _update_bond_list(bond_list, th_mesh, j,
                                            break_indices, a, b, eta)

            force, torque = _get_forces(bond_list, th_mesh, bond_max, nu, d)
            v = np.append(arr=v, values=v_f_interp(t_mesh[-1]) + force/eta_v)
            om = np.append(arr=om, values=om_f_interp(t_mesh[-1])
                                          + torque/eta_om)
            master_list = np.append(arr=master_list, values=bond_list)

        bond_counts = np.array([bond_list.shape[0]
                                for bond_list in master_list])
        avg_lengths = None
    else:
        raise Exception('master_list has an invalid number of dimensions')
    return (master_list, bond_counts, v, om, force_arr, torque_arr, t_mesh,
            avg_lengths)


def pde_eulerian(M, N, time_steps, init, bond_max, scheme='bw', **kwargs):
    """ Solves the full eulerian PDE model """

    # Define the problem parameters
    (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history) = set_parameters(**kwargs)

    v_f, om_f = _handle_velocities(v_f, om_f)

    # Define coordinate meshes and mesh widths
    z_mesh, th_mesh, l_mesh, t_mesh, h, nu, dt = (
        _generate_coordinate_arrays(M, N, time_steps, L, T, d, 'det')
    )

    # Check for the CFL condition
    _cfl_check(v_f, om_f, dt, h, nu, scheme)

    if init == 'free':
        m0 = np.zeros(shape=(2*M+1, N+1))
    elif init == 'tbound':
        eps = 0.1
        m0 = 1/(bond_max*N*eps**2)*np.exp(-(z_mesh[:, None]**2
                                            + th_mesh[None, :]**2)/eps**2)
    else:
        raise ValueError('init isn\'t a valid string')

    bond_mesh, v, om = _initialize_unknowns(v_f, om_f, m0, time_steps,
                                            save_bond_history)

    # Define distance/length mesh and formation/breaking rate meshes
    form_rate = on*kappa*np.exp(-eta/2*l_mesh**2)
    break_rate = off*np.exp(delta*l_mesh)

    bond_mesh, bond_counts, v, om, avg_lengths = _run_eulerian_model(
        bond_mesh, v, om, z_mesh, th_mesh, t_mesh, h, nu, dt, v_f, om_f, xi_v,
        xi_om, form_rate, break_rate, sat, d, scheme
    )

    force, torque = xi_v*(v - v_f), xi_om*(om - om_f)

    return (bond_mesh, bond_counts*bond_max*N/np.pi, v, om, avg_lengths, force,
            torque, t_mesh)


def rolling_ssa(M, N, time_steps, init, bond_max, correct_flux, **kwargs):
    """ Runs a single stochastic rolling simulation """

    # Define the problem parameters
    (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history) = set_parameters(**kwargs)

    v_f, om_f = _handle_velocities(v_f, om_f)

    # Define the theta bin centers and bin width. The slice below
    # selects only the th_mesh and nu outputs from the
    # _generate_coordinate_arrays method.

    th_mesh, t_uniform, nu = (
        _generate_coordinate_arrays(M, N, time_steps, L, T, d, 'sto')[1::2]
    )

    if init == 'free':
        m0 = np.zeros(shape=(0, 2))
    elif init == 'tbound':
        m0 = np.array([[0, np.searchsorted(th_mesh, 0).astype('float64')]])
    else:
        raise ValueError('init isn\'t a valid string')

    master_list, v_list, om_list, t_list = (
        _initialize_unknown_lists(v_f, om_f, m0, save_bond_history)
    )

    def coeffs_and_bounds(bins):
        coeffs = kappa*np.exp(-eta/2*(1 - np.cos(bins) + d)**2)*np.sqrt(np.pi/(2*eta))*(
            erf(np.sqrt(eta/2)*(np.sin(bins) + L)) - erf(np.sqrt(eta/2)*(np.sin(bins) - L))
        )
        coeffs = (bins > -np.pi/2)*(bins < np.pi/2)*coeffs
        a = (-L - np.sin(bins))/np.sqrt(1/eta)
        b = (L - np.sin(bins))/np.sqrt(1/eta)
        return coeffs, a, b

    max_step_length = T/time_steps*10
    (master_list, bond_counts, v_list, om_list, force_list, torque_list,
     t_list, avg_lengths) = (_run_stochastic_model(
        master_list, v_list, om_list, t_list, th_mesh, L, T, nu, bond_max, d,
        v_f, om_f, xi_v, xi_om, eta, delta, on, off, sat, correct_flux,
        max_step_length, coeffs_and_bounds
        )
    )

    return (master_list, bond_counts, v_list, om_list, force_list, torque_list,
            t_list, avg_lengths)


def count_variable(M, N, t_sample, time_steps, init, bond_max, correct_flux,
                   k=None, trials=None, **kwargs):
        start = timer()
        np.random.seed()
        (master_list, bond_counts, v_list, om_list, force_list, torque_list,
         t_list, avg_lengths) = rolling_ssa(M, N, time_steps, init, bond_max,
                                            correct_flux, **kwargs)

        indices = np.searchsorted(t_list, t_sample, side='left')
        end = timer()

        if k is not None and trials is not None:
            print('Completed {:d} of {:d} variable runs. This run took {:g}'
                  'seconds.'.format(k+1, trials,
                                    end-start))
        elif k is not None:
            print('Completed {:d} variable runs so far. This run took {:g}'
                  'seconds.'.format(k+1, end-start))
        elif trials is not None:
            print('Completed one of {:d} variable runs. This run took {:g}'
                  'seconds.'.format(trials, end-start))
        else:
            print('Completed one variable run. This run took {:g} seconds.'
                  .format(end-start))

        return (bond_counts[indices], v_list[indices], om_list[indices],
                force_list[indices], torque_list[indices],
                avg_lengths[indices[1:]-1], master_list, t_list)


def stochastic_experiments(trials, proc, M, N, time_steps, init, bond_max,
                           correct_flux, **kwargs):
    """ Run many stochastic experiments at once """

    assert type(proc) is int, 'proc must be a positive integer'

    T = set_parameters(**kwargs)[12]
    t_sample = np.linspace(0, T, time_steps+1)

    if proc == 1:
        result = [count_variable(M, N, t_sample, time_steps, init, bond_max,
                                 correct_flux, k, trials)
                  for k in range(trials)]
    elif proc > 1:
        pool = mp.Pool(processes=proc)
        result = [
            pool.apply_async(count_variable,
                             args=(M, N, t_sample, time_steps, init, bond_max,
                                   correct_flux, k, trials),
                             kwds=kwargs)
            for k in range(trials)
        ]

        result = [res.get() for res in result]
    else:
        raise ValueError('proc must be a positive integer')

    bond_counts = [res[0] for res in result]
    v_list = [res[1] for res in result]
    om_list = [res[2] for res in result]
    force_list = [res[3] for res in result]
    torque_list = [res[4] for res in result]
    length_list = [res[5] for res in result]
    master_list = [res[6] for res in result]
    t_list = [res[7] for res in result]

    count_array = np.vstack(bond_counts)
    v_array = np.vstack(v_list)
    om_array = np.vstack(om_list)
    force_array = np.vstack(force_list)
    torque_array = np.vstack(torque_list)
    avg_lengths = np.vstack(length_list)

    return (count_array, v_array, om_array, force_array, torque_array,
            avg_lengths, master_list, t_sample, t_list)


def generate_file_string(alg, M, N, time_steps, init, **kwargs):
    """ Generate the file string for the given parameters """

    (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
     save_bond_history) = set_parameters(**kwargs)

    if alg == 'det':
        file_path = '../data/deterministic/'
        scheme = kwargs['scheme']

        file_str = (
            'alg{0:s}_M{1:d}_N{2:d}_tsteps{3:d}_init{4:s}_scheme{5:s}_vf{6:g}_'
            'omf{7:g}_kappa{8:g}_eta{9:g}_d{10:g}_delta{11:g}_on{12:b}_'
            'off{13:b}_sat{14:b}_xiv{15:g}_xiom{16:g}_L{17:g}_T{18:g}_'
            'sbh{19:b}.npz'.format(alg, M, N, time_steps, init, scheme, v_f,
                                   om_f, kappa, eta, d, delta, on, off, sat,
                                   xi_v, xi_om, L, T, save_bond_history)
        )
    elif alg == 'sto':
        file_path = '../data/stochastic/'
        trials = kwargs['trials']
        bond_max = kwargs['bond_max']
        correct_flux = kwargs['correct_flux']

        file_str = (
            'alg{0:s}_M{1:d}_N{2:d}_tsteps{3:d}_init{4:s}_trials{5:d}_'
            'bmax{20:d}_cflux{21:b}_vf{6:g}_omf{7:g}_kappa{8:g}_eta{9:g}_'
            'd{10:g}_delta{11:g}_on{12:b}_off{13:b}_sat{14:b}_xiv{15:g}_'
            'xiom{16:g}_L{17:g}_T{18:g}_sbh{19:b}.npz'
                .format(alg, M, N, time_steps, init, trials, v_f, om_f, kappa,
                        eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
                        save_bond_history, bond_max, correct_flux)
        )
    else:
        raise ValueError('alg parameter is invalid')

    return file_path + 'expanded' + file_str


def write_deterministic_data(M, N, time_steps, init, bond_max, scheme,
                             **kwargs):
    """ Run deterministic simulation and write results to a file """

    file_path = generate_file_string('det', M, N, time_steps, init,
                                     bond_max=bond_max, scheme=scheme,
                                     **kwargs)

    try:
        with open(file_path, 'r') as file:
            print('The file {:s} already exists!'.format(file_path))
    except IOError:
        bond_mesh, bond_counts, v, om, avg_lengths, force, torque, t_mesh = (
            pde_eulerian(M, N, time_steps, init, bond_max, scheme, **kwargs)
        )
        np.savez_compressed(file_path, bond_counts, v, om, avg_lengths,
                            bond_mesh, force, torque, t_mesh,
                            bond_counts=bond_counts, v=v, om=om,
                            avg_lengths=avg_lengths, bond_mesh=bond_mesh,
                            force=force, torque=torque, t_mesh=t_mesh)
        print('Wrote output to {:s}'.format(file_path))
    return None


def load_deterministic_data(f):
    load_data = np.load(f)
    bond_counts, v, om, avg_lengths, bond_mesh, force, torque, t_mesh = (
        load_data['bond_counts'], load_data['v'], load_data['om'],
        load_data['avg_lengths'], load_data['bond_mesh'], load_data['force'],
        load_data['torque'], load_data['t_mesh']
    )
    print('Loaded file {:s}'.format(f))

    return bond_counts, v, om, avg_lengths, bond_mesh, force, torque, t_mesh


def write_stochastic_data(trials, proc, M, N, time_steps, init, bond_max,
                          correct_flux, **kwargs):
    """ Run stochastic simulations and write results to a file """

    file_path = generate_file_string(
        'sto', M, N, time_steps, init, trials=trials, bond_max=bond_max,
        correct_flux=correct_flux, **kwargs
    )

    try:
        with open(file_path, 'r') as file:
            print('The file {:s} already exists!'.format(file_path))
    except IOError:
        (count_array, v_array, om_array, force_array, torque_array,
         avg_lengths, master_list, t_sample) = (
            stochastic_experiments(trials, proc, M, N, time_steps, init,
                                   bond_max, correct_flux, **kwargs)[:-1]
        )
        np.savez_compressed(
            file_path, count_array, v_array, om_array, force_array,
            torque_array, avg_lengths, master_list, t_sample,
            count_array=count_array, v_array=v_array, om_array=om_array,
            force_array=force_array, torque_array=torque_array,
            avg_lengths=avg_lengths, master_list=master_list,
            t_sample=t_sample)
        print('Wrote output to {:s}'.format(file_path))


def load_stochastic_data(f):
    load_data = np.load(f)
    (count_array, v_array, om_array, force_array, torque_array, avg_lengths,
     master_list, t_sample) = (
        load_data['count_array'], load_data['v_array'], load_data['om_array'],
        load_data['force_array'], load_data['torque_array'],
        load_data['avg_lengths'], load_data['master_list'],
        load_data['t_sample']
    )
    print('Loaded file {:s}'.format(f))

    return (count_array, v_array, om_array, force_array, torque_array,
            avg_lengths, master_list, t_sample)


def _extract_means(stochastic_result):
    """ Extracts the mean of a set of stochastic experiment results """

    mean_results = tuple(np.mean(array, axis=0)
                         for array in stochastic_result[:-1])
    return mean_results + (stochastic_result[-1], )


def read_parameter_file(file_name):
    """ Reads a file containing the parameters to run """
    with open(file_name, 'r') as f:
        while True:
            command = f.readline().split()
            if len(command) < 1:
                continue

            if command[0] == 'v_f':
                gamma = float(command[1])
            elif command[0] == 'kappa':
                kappa = float(command[1])
            elif command[0] == 'eta':
                eta = float(command[1])
            elif command[0] == 'd':
                d = float(command[1])
            elif command[0] == 'delta':
                delta = float(command[1])
            elif command[0] == 'on':
                on = bool(command[1])
            elif command[0] == 'off':
                off = bool(command[1])
            elif command[0] == 'sat':
                sat = bool(command[1])
            elif command[0] == 'xi_v':
                xi_v = float(command[1])
            elif command[0] == 'xi_om':
                xi_om = float(command[1])
            elif command[0] == 'L':
                L = float(command[1])
            elif command[0] == 'T':
                T = float(command[1])
            elif command[0] == 'bond_max':
                bond_max = int(command[1])
            elif command[0] == 'save_bond_history':
                save_bond_history = bool(command[1])
            elif command[0] == 'M':
                M = int(command[1])
            elif command[0] == 'N':
                N = int(command[1])
            elif command[0] == 'time_steps':
                time_steps = int(command[1])
            elif command[0] == 'init':
                init = command[1]
            elif command[0] == 'trials':
                trials = int(command[1])
            elif command[0] == 'proc':
                proc = int(command[1])
            elif command[0] == 'correct_flux':
                correct_flux = bool(command[1])
            elif command[0] == 'alg':
                alg = command[1]
            elif command[0] == 'file_name':
                file_name = command[1]
            elif command[0] == 'done':
                break
    return (gamma, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, L, T,
            bond_max, save_bond_history, M, N, time_steps, init, trials, proc,
            correct_flux, alg, file_name)


def main():
    """ Run the main function """


if __name__ == '__main__':
    M, N = 64, 64
    T = float(5)
    time_steps = int(5120*T)
    kappa = float(10)
    # m0 = np.zeros(shape=(2*M+1, N+1))
    init = 'free'
    # init = 'tbound'  # One bond between the platelet and surface
    model_outputs = []
    bond_max = 10
    trials = 16
    proc = 4
    correct_flux = False

    write_deterministic_data(M, N, time_steps, init, bond_max=bond_max,
                             scheme='bw', T=T, kappa=kappa)
    write_stochastic_data(trials, proc, M, N, time_steps, init, bond_max,
                          correct_flux, T=T, kappa=kappa)

    # count_variable(M, N, np.linspace(0, T, time_steps+1), time_steps, init,
    #                bond_max, correct_flux, T=T, kappa=kappa)

    # for scheme in ['up', 'bw']:
    #     model_outputs.append(pde_eulerian(M, N, time_steps, m0, scheme=scheme,
    #                                       save_bond_history=False)[1:])
    #
    # stochastic_results = (
    #     stochastic_experiments(trials, M, N, time_steps, init, bond_max,
    #                            correct_flux, save_bond_history=False)
    # )
    #
    # model_outputs.append(_extract_means(stochastic_results))
    #
    # fig, ax = plt.subplots(nrows=3, sharex='all', figsize=(6, 8))
    #
    # for i in range(3):
    #     bond_counts, v, om, t_mesh = model_outputs[i]
    #     ax[0].plot(t_mesh, v)
    #     ax[1].plot(t_mesh, om)
    #     ax[2].plot(t_mesh, bond_counts)
    # ax[2].set_xlabel('ND time ($s$)')
    # ax[0].set_ylabel('ND translation velocity ($v$)')
    # ax[1].set_ylabel('ND rotation rate ($\\omega$)')
    # ax[2].set_ylabel('ND bond quantity ($\\iint m$)')
    # plt.tight_layout()
    # plt.show()
