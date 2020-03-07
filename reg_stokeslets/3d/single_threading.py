import subprocess
import re
import sys
import os
import glob
import warnings
import ctypes

MKL = 'mkl'
OPENBLAS = 'openblas'


class BLAS:
    def __init__(self, cdll, kind):

        if kind not in (MKL, OPENBLAS):
            raise ValueError('kind must be {} or {}, got {} instead.'.format(
                MKL, OPENBLAS, kind))

        self.kind = kind
        self.cdll = cdll

        if kind == MKL:
            self.get_n_threads = cdll.MKL_Get_Max_Threads
            self.set_n_threads = cdll.MKL_Set_Num_Threads
        else:
            self.get_n_threads = cdll.openblas_get_num_threads
            self.set_n_threads = cdll.openblas_set_num_threads


def get_blas(numpy_module):
    LDD = 'ldd'
    LDD_PATTERN = r'^\t(?P<lib>.*{}.*) => (?P<path>.*) \(0x.*$'

    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, 'multiarray.*so'))[0]

    ldd_result = subprocess.check_output(
        args=[LDD, MULTIARRAY_PATH],
        universal_newlines=True
    )

    output = ldd_result

    if MKL in output:
        kind = MKL
    elif OPENBLAS in output:
        kind = OPENBLAS
    else:
        return None

    pattern = LDD_PATTERN.format(kind)
    match = re.search(pattern, output, flags=re.MULTILINE)

    if match:
        lib = ctypes.CDLL(match.groupdict()['path'])
        return BLAS(lib, kind)
    else:
        return None


class single_threaded:
    def __init__(self, numpy_module):
        self.blas = get_blas(numpy_module)

    def __enter__(self):
        if self.blas is not None:
            self.old_n_threads = self.blas.get_n_threads()
            self.blas.set_n_threads(1)
        else:
            warnings.warn(
                'No MKL/OpenBLAS found, assuming NumPy is single-threaded.'
            )

    def __exit__(self, *args):
        if self.blas is not None:
            self.blas.set_n_threads(self.old_n_threads)
            if self.blas.get_n_threads() != self.old_n_threads:
                message = (
                    'Failed to reset {} to {} threads (previous value).'
                    .format(self.blas.kind, self.old_n_threads)
                )
                raise RuntimeError(message)


if __name__ == '__main__':
    import numpy as np

    matrix = np.random.rand(100, 100)
    # this uses however many threads MKL/OpenBLAS uses
    result = np.linalg.svd(matrix)

    # this uses one thread
    with single_threaded(np):
        result = np.linalg.svd(matrix)
