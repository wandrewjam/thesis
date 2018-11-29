# The default biological/physical values for the rolling model

"""
All functions for the rolling model will use parameter values defined in
this script, unless overwritten by **kwargs.
"""

gamma = 20
kappa = 1
eta = .1
d = .1
delta = 3
on = True
off = True
sat = True
xi_v = .001
xi_om = .001
v_f = (1 + d)*gamma
om_f = gamma
