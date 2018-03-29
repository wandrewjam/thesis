from scipy import linspace, array
from scipy.integrate import odeint
from pylab import *
from matplotlib import rc
from constants import *

class PKA:
        
        ####cAMP-PKA
        t0 = 0.0001
        t1 = 1.0
        t2 = 0.067082
        t3 = 2.0
        t4 = 1.75
        t5 = 7.0
        t6 = 0.006
        t7 = 2.22222
        t8 = 0.0014
        t9 = 0.933333
        t10 = 0.8
        t11 = 1.59009
        t12 = 0.00016
        t13 = 0.106667
        t14 = 0.0018
        t15 = 0.666667
        t16 = 0.32
        t17 = 15.9009
        t18 = 0.0014
        ####

        def __init__(self, cAMP, C, R, R_C, R2_cAMP, R2_cAMP_C, RA_cAMP, RB_cAMP, RB_cAMP_C):
                self.camp, self.C, self.R, self.RC, self.R2A, self.R2AC, self.RAA, self.RBA, self.RBAC = \
                    cAMP, C, R, R_C, R2_cAMP, R2_cAMP_C, RA_cAMP, RB_cAMP, RB_cAMP_C

        def dotcAMP(self):
                return (self.t6*self.RAA - self.t7*self.camp*self.R + self.t8*self.RBA - self.t9*self.camp*self.R +
                        self.t12*self.R2A - self.t13*self.RAA*self.camp + self.t14*self.R2A -
                        self.t15*self.camp*self.RBA + self.t10*self.RBAC - self.t11*self.camp*self.RC +
                        self.t16*self.R2AC - self.t17*self.RBAC*self.camp)

        def dotC(self):
                return(self.t0*self.RC - self.t1*self.R*self.C + self.t2*self.RBAC - self.t3*self.C*self.RBA +
                       self.t4*self.R2AC - self.t5*self.C*self.R2A)

        def dotR(self):
                return(self.t0*self.RC - self.t1*self.R*self.C + self.t6*self.RAA - self.t7*self.camp*self.R +
                       self.t18*self.RBA - self.t9*self.camp*self.R)

        def dotR_C(self):
                return(-self.t0*self.RC + self.t1*self.R*self.C + self.t10*self.RBAC - self.t11*self.camp*self.RC)

        def dotR2_cAMP(self):
                return(-self.t12*self.R2A + self.t13*self.camp*self.RAA - self.t14*self.R2A +
                       self.t15*self.camp*self.RBA + self.t4*self.R2AC - self.t5*self.C*self.R2A)

        def dotR2_cAMP_C(self):
                return(-self.t4*self.R2AC + self.t5*self.R2A*self.C - self.t16*self.R2AC + self.t17*self.camp*self.RBAC)

        def dotRA_cAMP(self):
                return(-self.t6*self.RAA + self.t7*self.camp*self.R + self.t12*self.R2A - self.t13*self.RAA*self.camp)

        def dotRB_cAMP(self):
                return(-self.t8*self.RBA + self.t9*self.camp*self.R + self.t14*self.R2A -
                       self.t15*self.RBA*self.camp + self.t2*self.RBAC - self.t3*self.C*self.RBA)

        def dotRB_cAMP_C(self):
                return(-self.t2*self.RBAC + self.t3*self.C*self.RBA - self.t10*self.RBAC +
                       self.t11*self.RC*self.camp + self.t16*self.R2AC - self.t17*self.RBAC*self.camp)
                                

if __name__ == "__main__":

    def dotu(u, t):
        par = PKA(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8])
        
                        
        dotcAMP = par.dotcAMP()
        dotC = par.dotC()
        dotR = par.dotR()
        dotR_C = par.dotR_C()
        dotR2_cAMP = par.dotR2_cAMP()
        dotR2_cAMP_C = par.dotR2_cAMP_C()
        dotRA_cAMP = par.dotRA_cAMP()
        dotRB_cAMP = par.dotRB_cAMP()
        dotRB_cAMP_C = par.dotRB_cAMP_C()
                
        return [dotcAMP, dotC, dotR, dotR_C, dotR2_cAMP, dotR2_cAMP_C, dotRA_cAMP, dotRB_cAMP, dotRB_cAMP_C]
        
        
    t = linspace(0, 150, 1e5)
                #i = 0.0
    u0 = [4, 0, 0, 1.3, 0, 0, 0, 0, 0]
    u = odeint(dotu, u0, t, mxstep = 10000)
    u = array(u).transpose()
    grid()
                #xlim(95,100)
    plot(t,u[1])
    
    show()
