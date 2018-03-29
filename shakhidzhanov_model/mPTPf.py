from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *

class mPTP:
    R = 1.98  # cal/(mol*K)
    F = 23050.0  # cal/(V*mol)
    T = 310.0  # K
    psi_0 = 0.11  # V Pokhilko2006152
    psi_1 = 0.02  # V Pokhilko2006152
    psi_2 = 0.07  # V Pokhilko2006152
    k_op = 0.034  # sec-1 Pokhilko2006152
    k_cp = 1.67  # sec-1 Pokhilko2006152
    P_ptp = 1.6  # mu l/(sec mg) Pokhilko2006152
    k_m = 0.032  # \mu M Pokhilko2006152
    H = 0.025  # \mu M Pokhilko2006152
    n = 4.0  # 8.0
    K = 10.0  # \mu M 150.0

    def __init__(self, Ca_cyt, Ca_mit, mPTP_open, Delpa_psi):
        self.c, self.cy, self.ptp, self.delta_psi = Ca_mit, Ca_cyt, mPTP_open, Delpa_psi

    def dotTop(self):
        if self.delta_psi >= self.psi_2:
            return self.k_op*((self.c**self.n)/((self.K**self.n)+(self.c**self.n))) * \
                   exp((self.psi_0 - self.delta_psi)/self.psi_1)*(1 - self.ptp) - \
                   self.k_cp*self.ptp*(self.H/(self.H + self.k_m))
        else:
            return self.k_op*((self.c**self.n)/((self.K**self.n)+(self.c**self.n))) * \
                   exp((self.psi_0 - self.psi_2)/self.psi_1)*(1 - self.ptp) - \
                   self.k_cp*self.ptp*(self.H/(self.H + self.k_m))

    def dotCa_cyt(self):
        return (self.P_ptp*self.ptp*(self.c - self.cy))/v_cyt

    def dotCa_mit(self):
        return -(self.P_ptp*self.ptp*(self.c - self.cy))/v_mit


if __name__ == "__main__":
    def dotu(u, t):
        PTP = mPTP(u[0], u[1], u[2], 0.19)
        dotCa_cyt = PTP.dotCa_cyt()
        dotCa_mit = PTP.dotCa_mit()
        dotTop = PTP.dotTop()
        p.append(u[2])
        return [dotCa_cyt, dotCa_mit, dotTop]

    p = []
    g = []
    l = []

    t = linspace(0, 100, 1e4)
    i = 0.0
    while i <= 400.0:
        u0 = [0.0, i, 0]
        u = odeint(dotu, u0, t)
        l.append(max(p))
        g.append(i)
        i = i + 1.0

    grid()
    xlabel('Ca2+_mit [uM]')
    ylabel('% open MP')
    plot(g, l)
    show()
