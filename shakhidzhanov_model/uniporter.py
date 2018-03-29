from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *

p = []
l = []
j = []

class Uniporter:
    R = 1.98 #cal/(mol*K)
    F = 23050.0 #cal/(V*mol)
    T = 310.0 #K
    delta_psi1 = 0.091 #V Magnus1997
    K_trans = 19.0 #\mu M Magnus1997
    K_act = 0.38 #\mu M Magnus1997
    n_a = 2.8 #1/1 Magnus1997
    L = 110.0 #1/1 Magnus1997
    J_max = 5.0 #\mu mol/g/sec Magnus1997
    
    def __init__(self, Ca_cyt, Delta_Psi):
        self.c, self.delta_psi = Ca_cyt, Delta_Psi
    def f1(self):
        return 2.0*self.F*(self.delta_psi - self.delta_psi1)/(self.R*self.T)
    def f2(self):
        return 1.0 + (self.c/self.K_trans)
    def f3(self):
        return 1.0 + (self.c/self.K_act)
    def dotCa_mit(self):
        return (self.J_max*(self.c/self.K_trans)*(self.f2()**3)*self.f1()/((self.f2()**4 + self.L/(self.f3()**self.n_a))*(1 - exp(-self.f1()))))/v_mit
    def dotCa_cyt(self):
        return (-self.J_max*(self.c/self.K_trans)*(self.f2()**3)*self.f1()/((self.f2()**4 + self.L/(self.f3()**self.n_a))*(1 - exp(-self.f1()))))/v_cyt

if __name__ == "__main__":
    def dotu(u, t):
        Uni = Uniporter(u[0], 0.19)
        dotCa_cyt = Uni.dotCa_cyt()
        dotCa_mit = Uni.dotCa_mit()
        p.append(dotCa_mit)
        return [dotCa_cyt, dotCa_mit]
    i = 0.0
    t = linspace(0, 200, 1e4)
    while i<=6.0:
        u0 = [i, 0.0]
        u = odeint(dotu, u0, t)
        l.append(max(p))
        j.append(i)
        i = i + 0.1
    grid()
    xlabel('Ca2+ [uM]')
    ylabel('Ca2+ uptake umol/sec/g')
    plot(j, l)
    show()
