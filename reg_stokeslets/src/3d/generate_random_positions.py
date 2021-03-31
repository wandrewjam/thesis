import numpy as np
from run_binding_expts import pick_random_height
from motion_integration import find_min_separation
import os


def main():
    num_positions = 512

    saved_positions = np.empty(shape=(num_positions, 4))
    for i in range(num_positions):
        height = pick_random_height()
        while True:
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(0, np.pi)

            cp, sp = np.cos(phi), np.sin(phi)
            ct, st = np.cos(theta), np.sin(theta)
            e_m = np.array([cp, ct * sp, st * sp])
            sep = find_min_separation(height, e_m)

            if sep > 0.0154:
                break

        saved_positions[i, 0] = height
        saved_positions[i, 1:] = e_m

    save_file = os.path.expanduser('~/thesis/reg_stokeslets/src/3d/r_pos.npy')
    np.save(save_file, saved_positions)


if __name__ == '__main__':
    main()
