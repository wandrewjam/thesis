# The default parameter values for the rolling model

"""
All functions for the rolling model will use parameter values defined in
this script, unless overwritten by **kwargs.
"""

from warnings import warn

# Define a dictionary for biological/physical parameters.
biological_parameters = {
    'gamma': 20,
    'kappa': 1,
    'eta': .1,
    'd': .1,
    'delta': 3,
    'on': True,
    'off': True,
    'sat': True,
    'xi_v': .01,
    'xi_om': .01
}

# Define a dictionary for numerical parameters.
numerical_parameters = {
    'L': 2.5,
    'T': 1,
    'bond_max': 10,
    'save_bond_history': False
}

# Merge the two dictionaries.
parameters = biological_parameters.copy()
parameters.update(numerical_parameters)


def set_parameters(**kwargs):
    """
    Given user-specified parameter values, merge with the default
    parameters and return parameters as individual variables.
    """

    parameters.update(kwargs)

    kappa = parameters['kappa']
    eta = parameters['eta']
    d = parameters['d']
    delta = parameters['delta']
    on = parameters['on']
    off = parameters['off']
    sat = parameters['sat']
    xi_v = parameters['xi_v']
    xi_om = parameters['xi_om']
    L = parameters['L']
    T = parameters['T']
    save_bond_history = parameters['save_bond_history']

    # Define v_f and om_f based on whether they are explicitly
    # specified or not.
    if 'v_f' in parameters.keys() and 'om_f' in parameters.keys():
        v_f = parameters['v_f']
        om_f = parameters['om_f']
    else:
        if 'v_f' in parameters.keys() or 'om_f' in parameters.keys():
            warn('Only one of v_f or om_f is specified. These values '
                 'will be set using gamma instead.')
        v_f = parameters['gamma'] * (1 + d)
        om_f = parameters['gamma']

    return v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om, \
        L, T, save_bond_history
