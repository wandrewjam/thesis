#!/usr/bin/env pytnon
# vim: set fileencoding=utf-8 ts=4 sw=4 expandtab:

from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *

P0 = []
t0 = []


class InsP3Receptor:
    k1, km1, k2, km2, k3, km3, k4, km4 = 0.64, 0.04, 37.4, 1.4, 0.11, 29.8, 4, 0.54  # 1/(s*muM) Sneyd19022002
    L1, L3, L5 = 0.12, 0.025, 54.7  # muM Sneyd19022002
    l2, l4, l6, lm2, lm4, lm6 = 1.7, 1.7, 4707, 0.8, 2.5, 11.4  # 1/s and ( 1/(s*muM) for l4 and lm4) Sneyd19022002

    def __init__(self, IP3Rr, IP3Ro, IP3Ra, IP3Ri1, IP3Ri2, IP3Rs, IP3, Ca_cyt):
        self.r, self.o, self.a, self.i1, self.i2, self.s, self.ip3, self.c = IP3Rr, IP3Ro, IP3Ra, IP3Ri1, IP3Ri2, IP3Rs, IP3, Ca_cyt

    def f1(self):  # Функции из Sneyd19022002
        return (self.k1*self.L1 + self.l2)*self.c/(self.L1 + self.c*(1.0 + self.L1/self.L3))

    def f2(self):
        return (self.k2*self.L3 + self.l4*self.c)/(self.L3 + self.c*(1.0 + self.L3/self.L1))

    def fm2(self):
        return	(self.km2 + self.lm4*self.c)/(1.0 + self.c/self.L5)

    def f3(self):
        return	(self.k3*self.L5)/(self.L5 + self.c)

    def f4(self):
        return (self.k4*self.L5 + self.l6)*self.c/(self.L5 + self.c)

    def fm4(self):
        return self.L1*(self.km4 + self.lm6)/(self.L1 + self.c)

    def f5(self):
        return (self.k1*self.L1 + self.l2)*self.c/(self.L1 + self.c)

    def dotIP3r(self):  # Диффуры из Sneyd19022002
        return self.fm2()*self.o - self.f2()*self.ip3*self.r + (self.km1 + self.lm2)*self.i1 - self.f1()*self.r

    def dotIP3Ro(self):
        return self.f2()*self.ip3*self.r - (self.fm2() + self.f4() + self.f3())*self.o + self.fm4()*self.a + self.km3*self.s

    def dotIP3Ra(self):
        return self.f4()*self.o - self.fm4()*self.a - self.f5()*self.a + (self.km1 + self.lm2)*self.i2

    def dotIP3Ri1(self):
        return self.f1()*self.r - (self.km1 + self.lm2)*self.i1

    def dotIP3Ri2(self):
        return self.f5()*self.a - (self.km1 + self.lm2)*self.i2

    def dotIP3Rs(self):
        return self.f3()*self.o - self.km3*self.s

    def dotIP3(self):
        return self.fm2()*self.o - self.f2()*self.ip3*self.r


if __name__ == "__main__":
    Ca_cyt = 0.1  # muM

    def dotu(u, t):
        IP3R = InsP3Receptor(u[1], u[2], u[3], u[4], u[5], u[6], u[0], u[7])

        dotIP3 = IP3R.dotIP3()
        dotIP3Rr = IP3R.dotIP3r()
        dotIP3Ro = IP3R.dotIP3Ro()
        dotIP3Ra = IP3R.dotIP3Ra()
        dotIP3Ri1 = IP3R.dotIP3Ri1()
        dotIP3Ri2 = IP3R.dotIP3Ri2()
        dotIP3Rs = IP3R.dotIP3Rs()
        P0.append(((0.9*IP3R.a + 0.1*IP3R.o)/(IP3R.a + IP3R.o + IP3R.i1 + IP3R.i2 + IP3R.r + IP3R.s))**4.0)  # Вер откр канала
        t0.append(t)

        return [dotIP3, dotIP3Rr, dotIP3Ro, dotIP3Ra, dotIP3Ri1, dotIP3Ri2, dotIP3Rs, 0.0]

    y = [1.0, 0.2, 0.02]
    e = 0
    grid()
    xscale('log')
    xlabel('Ca [uM]')
    ylabel('Open Probability')
    while e <= 2:
        i = 0.0
        while i <= 100.0:
            t = linspace(0, 700, 1e4)
            u0 = [y[e], 4.1513*(10.0**(-4.0)), 0.0, 0.0, 0.0, 0.0, 0.0, i]
            u = odeint(dotu, u0, t)
            u = array(u).transpose()
            l.append(P0[-1])
            c.append(i)
            i = i+0.001
        e = e+1
        plot(c, l)
    show()
