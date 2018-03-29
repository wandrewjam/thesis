from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *

class P2Y12r:
    k1, kGact, kGTP = 7.9, 1.2, 0.2 #s^-1
    k3 = 96.0*10**(0) # 1/mu M s^-1
    Ka = 10.0**(-3)
    Kact = 10.0**(-1)
    Kg = 10.0**(-1)
    alpha, beta = 100.0, 100.0
    k11 = 10.0**(-3)
    kG = 10.0**(-1)

    def __init__(self, PAR1, PAR1_Tr, PAR1_Tr_G, PAR1a_Tr, PAR1a_Tr_G, GqGTP, GqGDP, Gqbg, G, Tr):
        self.r, self.r_l, self.r_l_g, self.ra_l, self.ra_l_g, self.gq_gtp, self.gq_gdp, self.g_bg, self.g, self.l = PAR1, PAR1_Tr, PAR1_Tr_G, PAR1a_Tr, PAR1a_Tr_G, GqGTP, GqGDP, Gqbg, G, Tr

    def dotP2Y12(self):
        return (- self.k3*self.r*self.l + (self.k3/self.Ka)*self.r_l)

    def dotP2Y12_ADP(self):
        return (self.k3*self.r*self.l - (self.k3/self.Ka)*self.r_l - self.k1*self.r_l + (self.k1/self.Kact)*self.ra_l - self.k11*self.r_l*self.g + (self.k11/(self.Kg))*self.r_l_g)

    def dotP2Y12_ADP_G(self):
        return (self.k11*self.r_l*self.g - (self.k11/(self.Kg))*self.r_l_g + (self.k1/self.Kact)*self.ra_l_g - self.k1*self.alpha*self.r_l_g)

    def dotP2Y12a_ADP(self):
        return (self.k1*self.r_l - (self.k1/self.Kact)*self.ra_l - self.beta*self.k11*self.ra_l*self.g + (self.k11/(self.Kg))*self.ra_l_g+self.kGact*self.ra_l_g)

    def dotP2Y12a_ADP_G(self):
        return (self.beta*self.k11*self.ra_l*self.g - (self.k11/(self.Kg))*self.ra_l_g  +   self.alpha*self.k1*self.r_l_g - (self.k1/self.Kact)*self.ra_l_g - self.kGact*self.ra_l_g)

    def dotGiaGTP(self):
        return (self.kGact*(self.ra_l_g) - self.kGTP*self.gq_gtp)

    def dotGiaGDP(self):
        return (self.kGTP*self.gq_gtp - self.kG*self.gq_gdp*self.g_bg)

    def dotGbg(self):
        return (self.kGact*(self.ra_l_g) - self.kG*self.gq_gdp*self.g_bg)

    def dotG(self):
        return (-self.k11*self.r_l*self.g + (self.k11/(self.Kg))*self.r_l_g - self.beta*self.k11*self.ra_l*self.g +
                (self.k11/(self.Kg))*self.ra_l_g + self.kG*self.gq_gdp*self.g_bg)


if __name__ == "__main__":
    i = 0.0
    def dotu(u, t):
        par = P2Y12r(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9])
        dotP2Y12 = par.dotP2Y12()
        dotP2Y12_ADP = par.dotP2Y12_ADP()
        dotP2Y12_ADP_G = par.dotP2Y12_ADP_G()
        dotP2Y12a_ADP = par.dotP2Y12a_ADP()
        dotP2Y12a_ADP_G = par.dotP2Y12a_ADP_G()
        dotGiaGTP = par.dotGiaGTP()
        dotGiaGDP = par.dotGiaGDP()
        dotGbg = par.dotGbg()
        dotG = par.dotG()
        return [dotP2Y12, dotP2Y12_ADP, dotP2Y12_ADP_G, dotP2Y12a_ADP, dotP2Y12a_ADP_G, dotGiaGTP, dotGiaGDP, dotGbg, dotG, 0.0]

    u0 = [0.3, 0, 0, 0, 0, 0, 0, 0, 0.3, 20000.0]
    t = linspace(0, 0.1, 1e5)
    u = odeint(dotu, u0, t, mxstep=10000)
    u = array(u).transpose()
    plot(t,u[0], label='P2Y12')
    plot(t,u[1], label='P2Y12-ADP')
    plot(t,u[2], label='P2Y12-ADP-G')
    plot(t,u[3], label='P2Y12*-ADP')
    plot(t,u[4], label='P2Y12*-ADP-G')
    legend()
    grid()
    show()
