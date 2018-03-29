#!/usr/bin/env pytnon
# vim: set fileencoding=utf-8 ts=4 sw=4 expandtab:

from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *

class SR_Ca_ATPase2b:
    k1, k3, k4, k5, k6 = 280.0, 600.0, 65.0, 250.0, 105.0  # 1/s Vandecaetsbeek03112009
    r1, r2_1, r2_2, r3, r4, r6 = 200.0, 160.0, 1.6, 60.0, 25.0, 1.6  # 1/s Vandecaetsbeek03112009
    k2_1, k2_2, r5 = 9.4*100, 900.0, 8.0*(10**(-5))  # l/(muM S) l/(muM S) l^2/(muM^2 s) Vandecaetsbeek03112009

    def __init__(self, Serca2bE1, Serca2bE1Ca, Serca2bE1CaCa, Serca2bE1CaCaP, Serca2bE2CaCaP, Serca2bE2P, Serca2bE2,
                 Ca_cyt, Ca_dts):
        self.e1, self.e1ca, self.e1caca, self.e1cacap, self.e2cacap, self.e2p, self.e2, self.c_cyt, self.c_dts = \
            Serca2bE1, Serca2bE1Ca, Serca2bE1CaCa, Serca2bE1CaCaP, Serca2bE2CaCaP, Serca2bE2P, Serca2bE2, Ca_cyt, Ca_dts

    def dotSerca2bE1(self):
        return (self.r2_1*self.e1ca - self.k2_1*self.c_cyt*self.e1 + self.k1*self.e2 - self.r1*self.e1)/v_cyt

    def dotSerca2bE1Ca(self):
        return (self.k2_1*self.c_cyt*self.e1 - self.r2_1*self.e1ca - self.k2_2*self.c_cyt*self.e1ca +
                self.r2_2*self.e1caca)/v_cyt

    def dotSerca2bE1CaCa(self):
        return (self.k2_2*self.c_cyt*self.e1ca - self.r2_2*self.e1caca - self.k3*self.e1caca +
                self.r3*self.e1cacap)/v_cyt

    def dotSerca2bE1CaCaP(self):
        return (self.k3*self.e1caca - self.r3*self.e1cacap - self.k4*self.e1cacap + self.r4*self.e2cacap)/v_cyt

    def dotSerca2bE2CaCaP(self):
        return (self.k4*self.e1cacap - self.r4*self.e2cacap - self.k5*self.e2cacap +
                self.r5*self.e2p*self.c_dts*self.c_dts)/v_dts

    def dotSerca2bE2P(self):
        return (self.k5*self.e2cacap - self.r5*self.e2p*self.c_dts*self.c_dts - self.k6*self.e2p + self.r6*self.e2)/v_dts

    def dotSerca2bE2(self):
        return (self.k6*self.e2p - self.r6*self.e2 - self.k1*self.e2 + self.r1*self.e1)/v_dts

    def dotCa_cyt(self):
        return (self.r2_1*self.e1ca - self.k2_1*self.c_cyt*self.e1 - self.k2_2*self.c_cyt*self.e1ca +
                self.r2_2*self.e1caca)/v_cyt

    def dotCa_dts(self):
        return 2*(self.k5*self.e2cacap - self.r5*self.e2p*self.c_dts*self.c_dts)/v_dts


if __name__ == "__main__":
    g = []
    l = []
    f = []
    i = 0.0

    def dotu(u, t):
        Serca2b = SR_Ca_ATPase2b(u[0], u[1], u[2], u[3], u[4], u[5], u[6], i, 1300)

        dotSerca2bE1 = Serca2b.dotSerca2bE1()
        dotSerca2bE1Ca = Serca2b.dotSerca2bE1Ca()
        dotSerca2bE1CaCa = Serca2b.dotSerca2bE1CaCa()
        dotSerca2bE1CaCaP = Serca2b.dotSerca2bE1CaCaP()
        dotSerca2bE2CaCaP = Serca2b.dotSerca2bE2CaCaP()
        dotSerca2bE2P = Serca2b.dotSerca2bE2P()
        dotSerca2bE2 = Serca2b.dotSerca2bE2()
        dotCa_cyt = Serca2b.dotCa_cyt()
        dotCa_dts = Serca2b.dotCa_dts()
        g.append(dotCa_dts/2.0)

        return [dotSerca2bE1, dotSerca2bE1Ca, dotSerca2bE1CaCa, dotSerca2bE1CaCaP, dotSerca2bE2CaCaP, dotSerca2bE2P, dotSerca2bE2, dotCa_cyt, dotCa_dts]


    t = linspace(0, 0.5, 1e5)
    u0 = [0.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    i=0.0
    while i<20.0:
        u = odeint(dotu, u0, t, mxstep = 10000)
        u = array(u).transpose()
        f.append(i)
        print(i)
        i = i + 0.01
        l.append(g[-1])

    grid()
    xscale('log')
    ylim((0,25))
    plot(f, l, lw = 2)
    title('Serca2b')
    xlabel('Ca2+ free [uM]')
    ylabel('turnover rate [s-1]')
    show()
