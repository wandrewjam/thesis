import numpy as np


def sample_velocity(t, v, t_samp=None):
    if t_samp is None:
        t_samp = np.linspace(start=0, stop=t[-1], num=100)
        return v[np.searchsorted(t, t_samp)], t_samp
    else:
        return v[np.searchsorted(t, t_samp, side='right')-1]


def sample_position(t, v, t_samp=None):
    pos = np.cumsum(v[1:]*(t[1:] - t[:-1]))
    if t_samp is None:
        t_samp = np.linspace(start=0, stop=np.max(t), num=100)
        return np.insert(arr=pos[np.searchsorted(t, t_samp, side='right')-1], obj=0, values=0), t_samp
    else:
        return np.insert(arr=pos[np.searchsorted(t, t_samp, side='right')-1], obj=0, values=0)

