from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *

class PAR1:
    k1, kGact, kGTP = 7.9, 1.0, 1.0 #s^-1 10.1371/journal.pcbi.0030006
    k3 = 96.0 # 1/mu M s^-1 ligand + receptor 10.1371/journal.pcbi.0030006
    Ka = 10.0**(-1) #1/(muM) 10.1371/journal.pcbi.0030006
    Kact = 10.0**(-1) #1/1 #10.1371/journal.pcbi.0030006
    Kg = 10.0**(-2) #(number/cell)^-1 10.1371/journal.pcbi.0030006
    alpha, beta = 100, 100 #1/1 10.1371/journal.pcbi.0030006
    k11 = 10.0**(-3) #(number/cell^-1 s^-1 10.1371/journal.pcbi.0030006
    kG = 10.0**(-1) #(number/cell^-1 s^-1 10.1371/journal.pcbi.0030006
    cl = 2.0
        
    def __init__(self, PAR1, PAR1_Tr, PAR1_Tr_G, PAR1a_Tr, PAR1a_Tr_G, GqGTP, GqGDP, Gqbg, G, Tr):
        self.r, self.r_l, self.r_l_g, self.ra_l, self.ra_l_g, self.gq_gtp, self.gq_gdp, self.g_bg, self.g, self.l = PAR1, PAR1_Tr, PAR1_Tr_G, PAR1a_Tr, PAR1a_Tr_G, GqGTP, GqGDP, Gqbg, G, Tr

    def dotPAR1(self):
        return (- self.k3*self.r*self.l + (self.k3/self.Ka)*self.r_l)

    def dotPAR1_Tr(self):
        return (self.k3*self.r*self.l - (self.k3/self.Ka)*self.r_l - self.k1*self.r_l + (self.k1/self.Kact)*self.ra_l - self.k11*self.r_l*self.g + (self.k11/(self.Kg))*self.r_l_g)

    def dotPAR1_Tr_G(self):
        return (self.k11*self.r_l*self.g - (self.k11/(self.Kg))*self.r_l_g + (self.k1/self.Kact)*self.ra_l_g - self.k1*self.alpha*self.r_l_g)

    def dotPAR1a_Tr(self):
        return (self.k1*self.r_l - (self.k1/self.Kact)*self.ra_l - self.beta*self.k11*self.ra_l*self.g + (self.k11/(self.Kg))*self.ra_l_g+self.kGact*self.ra_l_g) - self.cl*self.ra_l

    def dotPAR1a_Tr_G(self):
        return (self.beta*self.k11*self.ra_l*self.g - (self.k11/(self.Kg))*self.ra_l_g  +   self.alpha*self.k1*self.r_l_g - (self.k1/self.Kact)*self.ra_l_g - self.kGact*self.ra_l_g)

    def dotGqGTP(self):
        return (self.kGact*(self.ra_l_g) - self.kGTP*self.gq_gtp)

    def dotGqGDP(self):
        return (self.kGTP*self.gq_gtp - self.kG*self.gq_gdp*self.g_bg)

    def dotGqbg(self):
        return (self.kGact*(self.ra_l_g) - self.kG*self.gq_gdp*self.g_bg)

    def dotG(self):
        return (-self.k11*self.r_l*self.g + (self.k11/(self.Kg))*self.r_l_g - self.beta*self.k11*self.ra_l*self.g + (self.k11/(self.Kg))*self.ra_l_g + self.kG*self.gq_gdp*self.g_bg)


if __name__ == "__main__":
    p = []
    l = []
    g = []

    def dotu(u, t):
        par = PAR1(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9])

        dotPAR1 = par.dotPAR1()
        dotPAR1_Tr = par.dotPAR1_Tr()
        dotPAR1_Tr_G = par.dotPAR1_Tr_G()
        dotPAR1a_Tr = par.dotPAR1a_Tr()
        dotPAR1a_Tr_G = par.dotPAR1a_Tr_G()
        dotGqGTP = par.dotGqGTP()
        dotGqGDP = par.dotGqGDP()
        dotGqbg = par.dotGqbg()
        dotG = par.dotG()
        p.append(u[5])

        return [dotPAR1, dotPAR1_Tr, dotPAR1_Tr_G, dotPAR1a_Tr, dotPAR1a_Tr_G, dotGqGTP, dotGqGDP, dotGqbg, dotG]


    t = linspace(0, 50, 1e5)
    i = 0.0
    grid()
    xlabel('t [sec]')
    ylabel('Gbg [uM]')
    while i<=0.5:
        u0 = [0.6, 0, 0, 0, 0, 0, 0, 0, 0.6, i]
        u = odeint(dotu, u0, t, mxstep = 10000)
        u = array(u).transpose()
        plot(t, u[7])
        print(i)
        i = i + 0.05
    show()
