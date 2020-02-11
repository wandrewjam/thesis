import numpy as np
from src.jv import delta_h, solve_pde


if __name__ == '__main__':
    eps1 = .43
    a = .13
    b, d = 1 - a, 1 - c
    h = 1. / N
    y = np.linspace(0, 1, num=N + 1)
    u_init = delta_h(y[1:], h)
    p0 = np.append(u_init, np.zeros(4 * N))

    max_step = 2 * h
    s_eval = np.arange(start=0, stop=s_max + max_step, step=max_step)
    u0_data, u1_data, v_data, f_data, vf_data = solve_pde(s_eval, p0, h=h, eps1=eps1, eps2=eps2, a=a, b=b, c=c, d=d, scheme=scheme)
