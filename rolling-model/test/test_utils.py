from unittest import TestCase
import numpy as np
from utils import length


class TestLength(TestCase):
    def setUp(self):
        self.arrayz = np.linspace(-1, 1, num=3)
        self.arrayth = np.linspace(-np.pi/2, np.pi/2, num=3)

    def test_zero_scalars(self):
        self.assertEqual(length(z=0, th=0, d=0), 0)

    def test_distance_scalars(self):
        self.assertEqual(length(z=0, th=0, d=.1), .1)

    def test_scalarz_arrayth(self):
        self.assertLessEqual(
            np.amax(np.abs(length(z=0, th=self.arrayth, d=0)
                           - np.array([np.sqrt(2), 0, np.sqrt(2)]))),
            2*np.finfo(float).eps
        )

    def test_arrayz_scalarth(self):
        self.assertEqual(
            np.amax(np.abs(length(z=self.arrayz, th=0, d=0)
                           - np.array([1, 0, 1]))), 0
        )

    def test_arrayz_arrayth(self):
        arrayth = np.linspace(-np.pi/2, np.pi/2, num=5)
        self.assertRaises(ValueError, length, self.arrayz, arrayth, 0)

class TestTorque(TestCase):
    def test_torque(self):
        self.assertEqual(0, 0)
