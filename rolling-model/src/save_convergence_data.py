import numpy as np
from time_dependent import time_dependent
import os.path
from timeit import default_timer as timer


def write_pde_data(M, N, time_steps, eta_v=0.01, gamma=20.0):
    file_str = './data/PDEdata_M{:d}_N{:d}_tsteps{:d}_eta{:g}_gamma{:g}_1order.npz'.format(
        M, N, time_steps, eta_v, gamma)

    if os.path.isfile(file_str):
        choice = input('Do you want to overwrite the current file? y or n ')
        while choice != 'n' and choice != 'y':
            choice = input('Please enter y or n. Do you want to overwrite the current file? y or n ')

        if choice == 'n':
            return

    start = timer()
    v, om, m_mesh, t = time_dependent(M=M, N=N, time_steps=time_steps, eta_v=eta_v, eta_om=eta_v, gamma=gamma,
                                      save_m=True)
    end = timer()

    print('It took {:g} seconds to solve the deterministic problem for the current parameters'.
          format(end - start))

    start = timer()
    np.savez_compressed(file_str, v, om, m_mesh, t, v=v, om=om, m_mesh=m_mesh, t=t)
    end = timer()
    print('And {:g} seconds to compress and save the data'.format(end-start))


if __name__ == '__main__':
    top = 7
    bottom = 2
    for j in range(bottom, top+1):
        write_pde_data(M=2**j, N=2**j, time_steps=25*2**j, gamma=0)
    print('Done!')