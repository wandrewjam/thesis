from sphere_integration_utils import (geom_weights, wall_stokeslet_integrand,
                                      generate_grid, sphere_integrate, phi)
from force_test import generate_stokeslet
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
        for i in range(6):
            assert np.linalg.norm(
                geom_weights(xi[:, np.newaxis], eta[np.newaxis, :],
                             patch=i + 1) - expected_result
            ) < 10*np.finfo(float).eps


class TestGridClass(object):
    def test_generate_grid(self):
        for n_nodes in [2, 4, 8, 16, 32]:
            xi_mesh, eta_mesh, cart_nodes = generate_grid(n_nodes)[:3]
            assert xi_mesh.shape == (n_nodes+1,)
            assert eta_mesh.shape == (n_nodes+1,)
            assert cart_nodes.shape == (6 * n_nodes**2 + 2, 3)

            for (a, b) in [(1.5, .5), (1., 1.), (.5, 1.5)]:
                xi_mesh, eta_mesh, cart_nodes = generate_grid(n_nodes, a=a,
                                                              b=b)[:3]
                assert np.linalg.norm(
                    np.sum(np.array([a, a, b]) ** (-2) * cart_nodes ** 2,
                           axis=-1) - np.ones(shape=6*n_nodes**2 + 2)
                ) < 1000 * n_nodes * np.finfo(float).eps


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

            integral = sphere_integrate(f, n_nodes)
            assert np.abs(integral) < n_nodes * np.finfo(float).eps

            # Also test the array version
            xi_mesh, eta_mesh, sphere_nodes, ind_map = generate_grid(n_nodes)
            integrand = f(sphere_nodes.T)
            integral = sphere_integrate(integrand, n_nodes)
            assert np.abs(integral) < n_nodes * np.finfo(float).eps

    def test_const_fun(self):
        def f(x_tuple):
            return np.ones(shape=x_tuple[0].shape)

        for (a, b) in [(1.5, .5), (1.25, 0.75), (1., 1.)]:
            for n_nodes in [2, 4, 8, 16]:
                integral = sphere_integrate(f, n_nodes, a=a, b=b)
                e = np.sqrt(np.abs(1 - b**2 / a**2))
                if e == 0:
                    exact = 4 * np.pi * a**2
                else:
                    exact = 2 * np.pi * a**2 * (1 + b**2 / a**2
                                                * np.arctanh(e) / e)
                assert np.abs(integral - exact) < 10. / (n_nodes ** 4)


class TestMeshGenerator(object):
    def test_mesh_generator(self):
        n_nodes = 8
        xi_mesh, eta_mesh, sphere_nodes, ind_map = generate_grid(n_nodes)

        for i in range(6):
            mapped_nodes = np.array(
                phi(xi_mesh[:, np.newaxis], eta_mesh[np.newaxis, :], patch=i+1)
            ).transpose((1, 2, 0))

            diff = mapped_nodes - sphere_nodes[ind_map[..., i]]
            assert (np.linalg.norm(diff.flatten())
                    < n_nodes**2 * np.finfo(float).eps)


class TestRegularizedStokeslets(object):
    def test_wall_bounded_stokeslets(self):
        """Test that the wall-bounded Stokeslet vanishes on the wall"""
        eps = 0.1
        num_cases = 5
        test_x_cases = np.concatenate([np.zeros(shape=(num_cases, 1)),
                                       10*np.random.rand(num_cases, 2) - 5],
                                      axis=1)
        center_cases = 10*np.random.rand(num_cases, 3) - 5
        force_cases = 10*np.random.rand(num_cases, 3) - 5
        for test_x in test_x_cases:
            for center in center_cases:
                for force in force_cases:
                    result = wall_stokeslet_integrand(test_x, center, eps,
                                                      force)
                    assert np.linalg.norm(result) < np.finfo(float).eps

    def test_vectorized_stokeslet(self):
        """Test that the vectorized wall-bounded Stokeslet vanishes"""
        eps = 0.1
        x, y, z = [0, 1, 2], [-1, 0, 1], [-1, 0, 1]
        nodes = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        stokeslet = generate_stokeslet(eps, nodes, type='wall', vectorized=True)
        assert np.all(np.abs(
            stokeslet[np.ix_(nodes[:, 0] == 0, nodes[:, 0] != 0)]
        ) < np.finfo(float).eps)
        assert np.all(np.abs(
            stokeslet[np.ix_(nodes[:, 0] != 0, nodes[:, 0] == 0)]
        ) < np.finfo(float).eps)
        # A stokeslet located on the wall and evaluated at the same
        # point is nan
        assert np.all(np.isnan(stokeslet[nodes[:, 0] == 0, nodes[:, 0] == 0]))
