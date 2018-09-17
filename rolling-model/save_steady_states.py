from steady_state import steady_state_sweep
import numpy as np

d_prime = 0.1
om_number, v_number = 11, 11
eta_om, eta_v = 0.0001, 0.0001
sat = True

v_f, om_f, vees, omegas, forces, torques = \
    steady_state_sweep(d_prime=d_prime, om_number=om_number, v_number=v_number,
                       eta_om=eta_om, eta_v=eta_v, saturation=sat, plot=False)

np.savez('ss_sweep_dprime{:g}_num{:g}_sat{:b}'.format(d_prime, om_number, sat),
         v_f=v_f, om_f=om_f, vees=vees, omegas=omegas, forces=forces, torques=torques)
print('Done!')
