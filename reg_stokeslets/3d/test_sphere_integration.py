from sphere_integration import phi
import numpy as np
import pytest


def test_phi():
    assert phi(0, 0, 1) == (1, 0, 0)


def test_phi1():
    with pytest.raises(AssertionError):
        phi(0, 0, 's')


def test_phi2():
    with pytest.raises(ValueError):
        phi(0, 0, 0)


def test_phi3():
    with pytest.raises(ValueError):
        phi(0, 0, 7)


def test_phi4():
    results = [(1, 0, 0), (0, 1, 0), (-1, 0, 0),
               (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for i in range(6):
        patch = i + 1
        assert phi(0, 0, patch) == results[i]


def test_phi5():
    xi = np.linspace(-np.pi / 4, np.pi / 4, num=3)
    eta = 0
    expected_result = (np.array([1 / np.sqrt(2), 1, 1 / np.sqrt(2)]),
                       np.array([-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]),
                       np.zeros(3))
    actual_result = phi(xi, eta, 1)
    assert (np.linalg.norm(np.array(actual_result) - np.array(expected_result))
            < np.finfo(float).eps)


def test_phi6():
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


def test_geom_weights():
    assert False
