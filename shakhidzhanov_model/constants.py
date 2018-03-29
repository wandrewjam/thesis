from pylab import *

#####Statistics############################################################################
Sigma = 0.1  # muM
###########################################################################################

#####PAR1R#################################################################################
Tr_0 = 0.0  # muM
PAR1_0 = G_0 = 0.8  # muM
PAR1_Tr_0 = 0.0  # muM
PAR1_Tr_G_0 = 0.0  # muM
PAR1a_Tr_0 = 0.0  # muM
PAR1a_Tr_G_0 = 0.0  # muM
GqGTP_0 = 0.0  # muM
GqGDP_0 = 0.0  # muM
Gqbg_0 = 0.0  # muM
###########################################################################################

#####P2Y12#################################################################################
ADP_0 = 200.0  # muM
P2Y12_0 = G1_0 = 0.3  # standard 0.3 muM
P2Y12_ADP_0 = 0.0  # muM
P2Y12_ADP_G_0 = 0.0  # muM
P2Y12a_ADP_0 = 0.0  # muM
P2Y12a_ADP_G_0 = 0.0  # muM
GiaGTP_0 = 0.0  # muM
GiaGDP_0 = 0.0  # muM
Gbg_0 = 0.0  # muM
###########################################################################################

#####cAMP##################################################################################
cAMP_0 = 4.0  # muM
AMP_0 = 0.0  # muM
PDE2_0 = 63.5  # muM
PDE3_0 = 222.7  # muM
PDE2a_0 = 0.5  # muM
PDE3a_0 = 2.3  # muM
ATP_0 = 1500.0  # muM
AC_0 = 1.0  # muM
###########################################################################################

#####PKA###################################################################################
C_0 = 0.0  # muM
R_0 = 0.0  # muM
R_C_0 = 1.3  # muM
R2_cAMP_0 = 0.0  # muM
R2_cAMP_C_0 = 0.0  # muM
RA_cAMP_0 = 0.0  # muM
RB_cAMP_0 = 0.0  # muM
RB_cAMP_C_0 = 0.0  # muM
###########################################################################################

#####Ca2+##################################################################################
Ca_cyt_0 = 0.005  # muM
Ca_dts_0 = 1300.0  # muM
Ca_mit_0 = 1.0  # muM
###########################################################################################

#####Serca2b###############################################################################
Serca2bE1_0 = 0.05  # muM Burkhart11102012
Serca2bE1Ca_0 = 0.0  # muM
Serca2bE1CaCa_0 = 0.0  # muM
Serca2bE1CaCaP_0 = 0.0  # muM
Serca2bE2CaCaP_0 = 0.0  # muM
Serca2bE2P_0 = 0.0  # muM
Serca2bE2_0 = 0.0  # muM
###########################################################################################

#####IP3###################################################################################
IP3r_0 = 0.12  # muM Burkhart11102012
IP3_0 = 0.0  # muM
IP3Ro_0 = 0.0  # muM
IP3Ra_0 = 0.0  # muM
IP3Ri1_0 = 0.0  # muM
IP3Ri2_0 = 0.0  # muM
IP3Rs_0 = 0.0  # muM
###########################################################################################

#####Mitochondria##########################################################################
Del_psi_0 = 0.19  # V
Top_0 = 0.0  # %
N_ant = 0.0040573  # 0.005
###########################################################################################

#####PIP2##################################################################################
PIP2_0 = 2.0  # muM doi:10.1146/annurev.biophys.31.082901.134259
###########################################################################################

#####Volume################################################################################
v_cyt = 4.0*(10**(-1))
v_dts = 8.0*(10**(-2))
v_mit = 4.0*(10**(-2))
###########################################################################################


def El_tr_ch(delta_psi):
    Const = 1.2*(10.0**(0))
    R = 1.98 #cal/(mol*K)
    F = 23050.0 #cal/(V*mol)
    T =  310.0 #K
    return Const*(exp(-F*delta_psi/(R*T)))


def NernstLeak(c_in, c_out, A, P0):
    R = 1.98  # cal/(mol*K)
    F = 23050.0  # cal/(V*mol)
    T =  310.0  # K
    z = 2.0  # Ca^(2)+
    return (R*T/(z*F))*log(c_in/c_out)*A*P0


def dotDelta_psi(J_uni, L_ca):
    C_mito = 1.45  # \mu mol/V g
    return (-2.0*J_uni + 2.0*L_ca)/C_mito
