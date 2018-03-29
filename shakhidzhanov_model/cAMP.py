from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *


class cAMP:
    # cAMP-PDE
    V1 = 120.0
    n = 1.0
    V2 = 0.1
    K = 3.0
    K1 = 50.0
    K2 = 0.2
    V3 = 0.7
    S = 120.0
    K4 = 1000.0
    k4 = 0.1
    # PDE2-PDE3
    k5 = 0.017
    # cAMP-PKA

    def __init__(self, cAMP, AMP, PDE2, PDE3, PDE2a, PDE3a, Gbg, AC, ATP):
        self.camp, self.amp, self.p2, self.p3, self.p2a, self.p3a, self.g_bg, self.ac, \
            self.atp = cAMP, AMP, PDE2, PDE3, PDE2a, PDE3a, Gbg, AC, ATP

    def dotcAMP(self):
        return (-(self.V1*self.camp*self.camp*self.p2a)/(self.K1 + self.camp*self.camp) -
                ((self.K + self.V2*self.camp)*self.camp*self.p3a)/(self.camp + self.K2) +
                self.atp*self.ac*(self.n - (self.V3*self.K4*self.g_bg)/(self.S + self.g_bg*self.K4))*self.k4)

    def dotAMP(self):
        return ((self.V1*self.camp*self.camp*self.p2a)/(self.K1 + self.camp*self.camp) +
                ((self.K + self.V2*self.camp)*self.camp*self.p3a)/(self.camp + self.K2))

    def dotPDE2(self):
        return -self.k5*self.p2 + self.k5*self.p2a

    def dotPDE3(self):
        return -self.k5*self.p3 + self.k5*self.p3a

    def dotPDE2a(self):
        return self.k5*self.p2 - self.k5*self.p2a

    def dotPDE3a(self):
        return self.k5*self.p3 - self.k5*self.p3a


if __name__ == "__main__":
    i = 0
    while i <= 0.2:
        def dotu(u, t):
            par = cAMP(u[0], u[1], u[2], u[3], u[4], u[5], i, 1, 1500)

            dotcAMP = par.dotcAMP()
            dotAMP = par.dotAMP()
            dotPDE2 = par.dotPDE2()
            dotPDE3 = par.dotPDE3()
            dotPDE2a = par.dotPDE2a()
            dotPDE3a = par.dotPDE3a()

            return [dotcAMP, dotAMP, dotPDE2, dotPDE3, dotPDE2a, dotPDE3a]


        t = linspace(0, 100, 10**5)
        # i = 0.0
        u0 = [4, 0, 63.5, 222.7, 0.5, 2.3]
        u = odeint(dotu, u0, t, mxstep=10000)
        u = array(u).transpose()
        grid()

        plot(t, u[0])
        i = i+0.01
    show()
