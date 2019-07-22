import numpy as np
from unittest import TestCase
from src.mle import change_vars


class TestChange_vars(TestCase):
    def setUp(self):
        self.model_pz = np.array([0.5, 1])
        self.fit_pz = np.zeros(2)

        self.model_pp = np.array([0.75, np.exp(1)])
        self.fit_pp = np.array([4./3, 1])

        self.model_pm = np.array([0.25, np.exp(-1)])
        self.fit_pm = np.array([-4./3, -1])

        self.model_all = np.stack([self.model_pm, self.model_pz,
                                   self.model_pp], axis=-1)
        self.fit_all = np.stack([self.fit_pm, self.fit_pz, self.fit_pp],
                                axis=-1)

    def test_2x1array(self):
        sol_dist = np.linalg.norm(change_vars(self.model_pz) - self.fit_pz)
        self.assertEqual(sol_dist, 0)

    def test_2x3arrayfwd(self):
        sol_dist = np.linalg.norm(change_vars(self.model_all) - self.fit_all)
        self.assertEqual(sol_dist, 0)

    def test_2x3arraybck(self):
        sol_dist = np.linalg.norm(change_vars(self.fit_all, forward=False)
                                  - self.model_all)
        self.assertAlmostEqual(sol_dist, 0, places=15)
