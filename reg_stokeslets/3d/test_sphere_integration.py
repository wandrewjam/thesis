from sphere_integration_utils import phi, geom_weights, generate_grid, sphere_integrate
import numpy as np
import pytest
from scipy.special import sph_harm


class TestPhiClass(object):
    def test_phi1(self):
        """Test an incorrect type of patch"""
        with pytest.raises(AssertionError):
            phi(0, 0, 's')

    def test_phi2(self):
        """Test that an error is raised when patch < 1"""
        with pytest.raises(ValueError):
            phi(0, 0, 0)

    def test_phi3(self):
        """Test that an error is raised when patch > 6"""
        with pytest.raises(ValueError):
            phi(0, 0, 7)

    def test_phi4(self):
        """Test the centers of each patch"""
        results = [(1, 0, 0), (0, 1, 0), (-1, 0, 0),
                   (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for i in range(6):
            patch = i + 1
            assert phi(0, 0, patch) == results[i]

    def test_phi5(self):
        """Test for broadcasting xi and eta arrays on patch 1"""
        xi = np.linspace(-np.pi / 4, np.pi / 4, num=3)
        eta = np.linspace(-np.pi / 4, np.pi / 4, num=3)

        expected_result = (
            np.array([[1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(3)],
                      [1 / np.sqrt(2), 1, 1 / np.sqrt(2)],
                      [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(3)]]),
            np.array([[-1 / np.sqrt(3), -1 / np.sqrt(2), -1 / np.sqrt(3)],
                      [0, 0, 0],
                      [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(3)]]),
            np.array([[-1 / np.sqrt(3), 0, 1 / np.sqrt(3)],
                      [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
                      [-1 / np.sqrt(3), 0, 1 / np.sqrt(3)]])
        )
        actual_result = phi(xi[:, np.newaxis], eta[np.newaxis, :], 1)

        assert (np.linalg.norm(np.array(actual_result) - np.array(expected_result))
                < 10 * np.finfo(float).eps)


class TestWeightsClass(object):
    def test_geom_weights(self):
        xi = np.linspace(-np.pi / 4, np.pi / 4, num=3)
        eta = np.linspace(-np.pi / 4, np.pi / 4, num=3)
        expected_result = np.zeros(shape=(3, 3))
        expected_result[1, 1] = 1
        expected_result[(0, 2, 1, 1), (1, 1, 0, 2)] = 1 / np.sqrt(2)
        expected_result[::2, ::2] = 4 / np.sqrt(3**3)
        print()
        assert np.linalg.norm(
            geom_weights(xi[:, np.newaxis], eta[np.newaxis, :])
            - expected_result) < 10*np.finfo(float).eps


class TestGridClass(object):
    def test_generate_grid(self):
        for n_nodes in [2, 4, 8, 16, 32]:
            xi_mesh, eta_mesh, sphere_nodes = generate_grid(n_nodes)
            assert xi_mesh.shape == (n_nodes+1,)
            assert eta_mesh.shape == (n_nodes+1,)
            assert sphere_nodes.shape == (6 * n_nodes**2 + 2, 3)
            assert np.linalg.norm(
                np.linalg.norm(sphere_nodes, axis=-1)
                - np.ones(shape=6*n_nodes**2 + 2)
            ) < n_nodes * np.finfo(float).eps


class TestSphereQuadrature(object):
    def test_spherical_harmonics(self):
        n_nodes = 8
        for i in range(8):
            m, n = 0, 2*i + 1

            def f(x_tuple):
                x, y, z = x_tuple
                theta = (np.arctan2(y, x)) % (2 * np.pi)
                pphi = np.arccos(z)
                return sph_harm(m, n, theta, pphi)

            integral = sphere_integrate(f, n_nodes, )
            assert np.abs(integral) < n_nodes * np.finfo(float).eps
