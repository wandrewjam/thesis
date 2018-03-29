from P2Y12 import *
from cAMP import *
from PKA import *
from PLC_P2Y12 import *
from IP3R import *
from SERCA2b import *
from uniporter import *
from antiporter import *
from mPTPf import *
from PLC import *
from PAR1r import *
from constants import *


def dotu(u, t):
    # Create instances of each class taking concentrations from the input u
    Serca2b = SR_Ca_ATPase2b(u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[0], u[1])
    IP3R = InsP3Receptor(u[11], u[12], u[13], u[14], u[15], u[16], u[10], u[0])
    Uni = Uniporter(u[0], u[18])
    Ant = Antiporter(u[2], u[18])
    PTP = mPTP(u[0], u[2], u[17], u[18])
    PLCb = PLC(u[10], u[24], u[0], PIP2_0)

    if t > 600.0:
        p2y12 = P2Y12r(u[28], u[29], u[30], u[31], u[32], u[33], u[34], u[35], u[36], u[52])

        if t > 600.0:
            par = PAR1(u[19], u[20], u[21], u[22], u[23], u[24], u[25], u[26], u[27], u[51])
        else:
            par = PAR1(u[19], u[20], u[21], u[22], u[23], u[24], u[25], u[26], u[27], 0.0)

    else:
        par = PAR1(u[19], u[20], u[21], u[22], u[23], u[24], u[25], u[26], u[27], 0.0)
        p2y12 = P2Y12r(u[28], u[29], u[30], u[31], u[32], u[33], u[34], u[35], u[36], 0.0)

    camp = cAMP(u[37], u[38], u[39], u[40], u[41], u[42], u[35], AC_0, ATP_0)
    pka = PKA(u[37], u[43], u[44], u[45], u[46], u[47], u[48], u[49], u[50])
    plc = PLC_P2Y12(u[4], PIP2_0, u[33], u[43], u[0])
    dotPAR1 = par.dotPAR1()
    dotPAR1_Tr = par.dotPAR1_Tr()
    dotPAR1_Tr_G = par.dotPAR1_Tr_G()
    dotPAR1a_Tr = par.dotPAR1a_Tr()
    dotPAR1a_Tr_G = par.dotPAR1a_Tr_G()
    dotGqGTP = par.dotGqGTP()
    dotGqGDP = par.dotGqGDP()
    dotGqbg = par.dotGqbg()
    dotG = par.dotG()
    dotSerca2bE1 = Serca2b.dotSerca2bE1()
    dotSerca2bE1Ca = Serca2b.dotSerca2bE1Ca()
    dotSerca2bE1CaCa = Serca2b.dotSerca2bE1CaCa()
    dotSerca2bE1CaCaP = Serca2b.dotSerca2bE1CaCaP()
    dotSerca2bE2CaCaP = Serca2b.dotSerca2bE2CaCaP()
    dotSerca2bE2P = Serca2b.dotSerca2bE2P()
    dotSerca2bE2 = Serca2b.dotSerca2bE2()
    dotIP3 = IP3R.dotIP3() + PLCb.dotIP3() + plc.dotIP3()
    dotIP3Rr = IP3R.dotIP3r()
    dotIP3Ro = IP3R.dotIP3Ro()
    dotIP3Ra = IP3R.dotIP3Ra()
    dotIP3Ri1 = IP3R.dotIP3Ri1()
    dotIP3Ri2 = IP3R.dotIP3Ri2()
    dotIP3Rs = IP3R.dotIP3Rs()
    dotTop = PTP.dotTop()
        
    P0 = (((0.9*IP3R.a + 0.1*IP3R.o)/(IP3R.a + IP3R.o + IP3R.i1 + IP3R.i2 + IP3R.r + IP3R.s))**4.0)
    s = IP3R.a + IP3R.o + IP3R.i1 + IP3R.i2 + IP3R.r + IP3R.s
        
    dotDel_psi = dotDelta_psi(Uni.dotCa_mit()*v_mit - El_tr_ch(u[18]), PTP.dotCa_mit()*v_mit)
    dotCa_cyt = Serca2b.dotCa_cyt() + NernstLeak(u[1], u[0], 3.5, 1.0)/v_cyt + \
                NernstLeak(u[1], u[0], 6.0**(5.0), P0*s)/v_cyt + Uni.dotCa_cyt() + \
                N_ant*Ant.dotCa_cyt() + PTP.dotCa_cyt()
    dotCa_dts = Serca2b.dotCa_dts() - NernstLeak(u[1], u[0], 3.5, 1.0)/v_dts - \
                NernstLeak(u[1], u[0], 6.0**(5.0), P0*s)/v_dts
    dotCa_mit = Uni.dotCa_mit() + N_ant*Ant.dotCa_mit() + PTP.dotCa_mit()
        
    dotP2Y12 = p2y12.dotP2Y12()
    dotP2Y12_ADP = p2y12.dotP2Y12_ADP()
    dotP2Y12_ADP_G = p2y12.dotP2Y12_ADP_G()
    dotP2Y12a_ADP = p2y12.dotP2Y12a_ADP()
    dotP2Y12a_ADP_G = p2y12.dotP2Y12a_ADP_G()
    dotGiaGTP = p2y12.dotGiaGTP()
    dotGiaGDP = p2y12.dotGiaGDP()
    dotGbg = p2y12.dotGbg()
    dotG1 = p2y12.dotG()
        
    dotAMP = camp.dotAMP()
    dotPDE2 = camp.dotPDE2()
    dotPDE3 = camp.dotPDE3()
    dotPDE2a = camp.dotPDE2a()
    dotPDE3a = camp.dotPDE3a()
        
    dotC = pka.dotC()
    dotR = pka.dotR()
    dotR_C = pka.dotR_C()
    dotR2_cAMP = pka.dotR2_cAMP()
    dotR2_cAMP_C = pka.dotR2_cAMP_C()
    dotRA_cAMP = pka.dotRA_cAMP()
    dotRB_cAMP = pka.dotRB_cAMP()
    dotRB_cAMP_C = pka.dotRB_cAMP_C()
    dotcAMP = camp.dotcAMP() + pka.dotcAMP()
        
    return[dotCa_cyt, dotCa_dts, dotCa_mit, dotSerca2bE1, dotSerca2bE1Ca, dotSerca2bE1CaCa, dotSerca2bE1CaCaP,  # 6
           dotSerca2bE2CaCaP, dotSerca2bE2P, dotSerca2bE2, dotIP3, dotIP3Rr, dotIP3Ro, dotIP3Ra, dotIP3Ri1, dotIP3Ri2,  # 15
           dotIP3Rs, dotTop, dotDel_psi, dotPAR1, dotPAR1_Tr, dotPAR1_Tr_G, dotPAR1a_Tr, dotPAR1a_Tr_G, dotGqGTP,  # 24
           dotGqGDP, dotGqbg, dotG, dotP2Y12, dotP2Y12_ADP, dotP2Y12_ADP_G, dotP2Y12a_ADP, dotP2Y12a_ADP_G, dotGiaGTP,  # 33
           dotGiaGDP, dotGbg, dotG1, dotcAMP, dotAMP, dotPDE2, dotPDE3, dotPDE2a, dotPDE3a, dotC, dotR, dotR_C,  # 45
           dotR2_cAMP, dotR2_cAMP_C, dotRA_cAMP, dotRB_cAMP, dotRB_cAMP_C, 0.0, 0.0]  # 52
