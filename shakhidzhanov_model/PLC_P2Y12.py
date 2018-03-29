from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *


class PLC_P2Y12:
    V_max = 14.0
    K_G = 2.0*(10**(-3))
    K_P = 84.0
    n1 = 2.0

    def __init__(self, IP3, PIP2, Gbg, C, Ca_cyt):
        self.ip3, self.pip2, self.g_bg, self.C, self.c = IP3, PIP2, Gbg, C, Ca_cyt

    def dotIP3(self):
        return 0.8*((1.0 - (5.3*self.C)/(2.0 + self.C))*(3.0*self.g_bg*self.V_max/(self.g_bg + self.K_G))*((self.pip2**self.n1)/((self.K_P**self.n1) + (self.pip2**self.n1))))/v_cyt
