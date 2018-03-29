from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *


class Antiporter:
    J_max = 0.42  # \mu mol/g/sec Magnus1997
    b = 0.0  # 1/1 Magnus1997
    k_Na = 9.4  # mM Magnus1997
    k_Ca = 0.003  # nmol/mg Magnus1997
    n = 2.0  # 1/1 Magnus1997
    C_Na = 30  # mM Magnus1997
    R = 1.98  # cal/(mol*K)
    F = 23050.0  # cal/(V*mol)
    T = 310.0  # K
    delta_psi1 = 0.091  # V Magnus1997

    def __init__(self, Ca_mit, Delta_Psi):
        self.c, self.delta_psi = Ca_mit, Delta_Psi

    def f1(self):
        return exp(self.b*self.F*(self.delta_psi - self.delta_psi1)/(self.R*self.T))

    def f2(self):
        return 1.0 + self.k_Na/self.C_Na

    def f3(self):
        return 1.0 + self.k_Ca/self.c

    def dotCa_mit(self):
        return (-self.J_max*self.f1()/((self.f2()**self.n)*self.f3()))/v_mit

    def dotCa_cyt(self):
        return (self.J_max*self.f1()/((self.f2()**self.n)*self.f3()))/v_cyt


if __name__ == "__main__":
    p = []
    g = []
    l = []

    def dotu(u, t):
        Ant = Antiporter(u[1], 0.19)
        dotCa_cyt = Ant.dotCa_cyt()
        dotCa_mit = Ant.dotCa_mit()
        p.append(dotCa_cyt)
        return [dotCa_cyt, dotCa_mit]

    t = linspace(0, 1000, 1e4)
    i=0.00001
    while i<=6.0:
        u0 = [0.0, i]
        u = odeint(dotu, u0, t)
        l.append(max(p))
        g.append(i)
        i=i+0.1
        print(i)

    grid()
    xlabel('Ca2+_mit [uM]')
    ylabel('Ca2+ uptake umol/sec/g')
    plot(g, l)
    show()
