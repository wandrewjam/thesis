# The default parameter values for the rolling model

"""
All functions for the rolling model will use parameter values defined in
this script, unless overwritten by **kwargs.
"""

from warnings import warn

# Define a dictionary for biological/physical parameters.
_biological_parameters = {
    'gamma': 40,
    'kappa': 1,
    'eta': 2.3e4,
    'd': .01,
    'delta': 1,
    'on': True,
    'off': True,
    'sat': True,
    'xi_v': 1e-6,
    'xi_om': 1e-6
}

# Define a dictionary for numerical parameters.
_numerical_parameters = {
    'L': 2.5,
    'T': 3,
    'bond_max': 10,
    'save_bond_history': False
}

# Merge the two dictionaries.
_parameters = _biological_parameters.copy()
_parameters.update(_numerical_parameters)


def set_parameters(**kwargs):
    """
    Given user-specified parameter values, merge with the default
    parameters and return parameters as individual variables.
    """

    _parameters.update(kwargs)

    kappa = _parameters['kappa']
    eta = _parameters['eta']
    d = _parameters['d']
    delta = _parameters['delta']
    on = _parameters['on']
    off = _parameters['off']
    sat = _parameters['sat']
    xi_v = _parameters['xi_v']
    xi_om = _parameters['xi_om']
    L = _parameters['L']
    T = _parameters['T']
    save_bond_history = _parameters['save_bond_history']

    # Define v_f and om_f based on whether they are explicitly
    # specified or not.
    if 'v_f' in _parameters.keys() and 'om_f' in _parameters.keys():
        v_f = _parameters['v_f']
        om_f = _parameters['om_f']
    else:
        if 'v_f' in _parameters.keys() or 'om_f' in _parameters.keys():
            warn('Only one of v_f or om_f is specified. These values '
                 'will be set using gamma instead.')
        v_f = _parameters['gamma'] * (1 + d)
        om_f = _parameters['gamma']/2.
        print('Set v_f and om_f using the shear rate')

    return (v_f, om_f, kappa, eta, d, delta, on, off, sat, xi_v, xi_om,
            L, T, save_bond_history)
