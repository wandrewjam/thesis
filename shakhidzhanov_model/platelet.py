#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sw=4 expandtab:

from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from diffeqsys import *        
from scipy import stats
from scipy.stats import norm

Thrombin = 10.0
ADP = 2500.0
Potential_number = 0.113996


def verfunc (delta_psi, Receptors, Thrombin):
    if delta_psi>=Potential_number:
        print('True', Receptors, Thrombin, (1.0-norm.cdf(Receptors, loc=PAR1_0, scale=Sigma))*100.0, delta_psi)
        return True
    else:
        print('False', Receptors, Thrombin, (1.0-norm.cdf(Receptors, loc=PAR1_0, scale=Sigma))*100.0, delta_psi)
        return False


def writefunc (delta_psi, Receptors, Thrombin):
    f = open('(Thrombin test2500).txt', 'a')
    f.write(str(Receptors) + ', \t' + str(Thrombin) + ', \t' + str((1.0-norm.cdf(Receptors, loc=PAR1_0, scale=Sigma)) *
                                                                   100.0) + ', \t' + str(delta_psi) + '\n')
    f.close()


def workfunc(Receptors):
    # Input: number of PAR1 receptors
    # Output: minimum of Delta Psi over the whole time interval
    t = linspace(0, 3000, 1e6)
    u0 = [Ca_cyt_0, Ca_dts_0, Ca_mit_0, Serca2bE1_0, Serca2bE1Ca_0, Serca2bE1CaCa_0, Serca2bE1CaCaP_0, Serca2bE2CaCaP_0,
          Serca2bE2P_0, Serca2bE2_0, IP3_0, IP3r_0, IP3Ro_0, IP3Ra_0, IP3Ri1_0, IP3Ri2_0, IP3Rs_0, Top_0, Del_psi_0,
          Receptors, PAR1_Tr_0, PAR1_Tr_G_0, PAR1a_Tr_0, PAR1a_Tr_G_0, GqGTP_0, GqGDP_0, Gqbg_0, G_0, P2Y12_0,
          P2Y12_ADP_0, P2Y12_ADP_G_0, P2Y12a_ADP_0,  P2Y12a_ADP_G_0, GiaGTP_0, GiaGDP_0, Gbg_0, G1_0, cAMP_0, AMP_0,
          PDE2_0, PDE3_0, PDE2a_0, PDE3a_0, C_0, R_0, R_C_0, R2_cAMP_0, R2_cAMP_C_0, RA_cAMP_0, RB_cAMP_0, RB_cAMP_C_0,
          Thrombin, ADP]
    u = odeint(dotu, u0, t)
    u = array(u).transpose()
    return min(u[18])


Receptors = ReceptorsPreviousStep = PAR1_0 + 0.4
ReceptorsStep = 0.05
ThrombinStep = 20.0
i_0 = 2.0
i_final = 32.0


def func(i, Receptors, Thrombin):
    u = workfunc(Receptors)
    i = i*2.0
    if i >= i_final:
        if verfunc(u, Receptors, Thrombin):
            writefunc(u, Receptors + ReceptorsStep/i, Thrombin)
        else:
            writefunc(u, Receptors - ReceptorsStep/i, Thrombin)
    else:
        if verfunc(u, Receptors, Thrombin):
            func(i, Receptors + ReceptorsStep/i,  Thrombin)
        else:
            func(i, Receptors - ReceptorsStep/i,  Thrombin)


while Thrombin <= 210.0:
    print('------------------------------------\n')
    Receptors = ReceptorsPreviousStep
    while Receptors > PAR1_0:
        u = workfunc(Receptors)
        if verfunc(u, Receptors, Thrombin):
            func(i_0, Receptors + ReceptorsStep/i_0, Thrombin)
            break
        ReceptorsPreviousStep = ReceptorsPreviousStep - ReceptorsStep
        Receptors = ReceptorsPreviousStep
    Thrombin = Thrombin + ThrombinStep




