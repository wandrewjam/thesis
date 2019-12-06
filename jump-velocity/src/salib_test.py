from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


if __name__ == '__main__':
    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359]]
    }

    param_values = saltelli.sample(problem, 1000)
    Y = Ishigami.evaluate(param_values)

    Si = sobol.analyze(problem, Y)
    print(Si['S1'])
