from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *


class PLC:
    V_max = 14.0  # 9.0 analysis
    K_G = 2.0*(10**(-3))  # ?
    K_P = 84.0  # muM analysis
    r5p = 0.05  # s-1 Ben-Jacob2009
    v3k = 2.0  # muM s-1 Ben-Jacob2009
    K_D = 0.7  # muM Ben-Jacob2009
    K3 = 1.0  # muM Ben-Jacob2009
    n = 2.0  # 1/1
    n2 = 4.0  # 1/1

    def __init__(self, IP3, Gaq, Ca_cyt, PIP2):
        self.ip3, self.g, self.c, self.pip2 = IP3, Gaq, Ca_cyt, PIP2

    def dotIP3(self):
        return ((self.V_max + 10.0*self.g*self.V_max/(self.g + self.K_G)) *
                ((self.pip2**self.n)/((self.K_P**self.n) + (self.pip2**self.n))) -
                self.r5p*self.ip3 - self.v3k*(self.c**self.n2)/((self.c**self.n2) + (self.K_D**self.n2)) *
                self.ip3/(self.ip3 + self.K3))/v_cyt


if __name__ == "__main__":
    p = []
    l = []
    g = []

    def dotu(u, t):
        PLCb = PLC(u[0], u[1], 0.05)
        dotIP3 = PLCb.dotIP3()
        p.append(u[0])
        return [dotIP3]

    t = linspace(0, 50, 1e5)
    grid()
    xlabel('Gq [uM]')
    ylabel('IP3 [uM]')
    Ga = 0
    while Ga<0.02:
        u0 = [0, Ga]
        u = odeint(dotu, u0, t, mxstep=10000)
        u = array(u).transpose()
        l.append(max(p))
        g.append(Ga)
        Ga = Ga + 0.0001
    plot(g, l)
    show()
