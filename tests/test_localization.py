import numpy as np

from src.localization import gaspari_cohn, cyclic_distance, localization_matrix


def test_gaspari_cohn_is_one_at_zero():
    assert np.isclose(gaspari_cohn(np.array([0.0]))[0], 1.0)


def test_gaspari_cohn_vanishes_beyond_two():
    z = np.array([2.0, 2.5, 10.0])
    assert np.allclose(gaspari_cohn(z), 0.0, atol=1e-10)


def test_gaspari_cohn_continuous_at_breakpoint_one():
    left = gaspari_cohn(np.array([0.999999]))[0]
    right = gaspari_cohn(np.array([1.000001]))[0]
    assert np.isclose(left, right, atol=1e-4)


def test_gaspari_cohn_symmetric_in_sign():
    z = np.array([0.3, 1.2, -0.3, -1.2])
    vals = gaspari_cohn(z)
    assert np.isclose(vals[0], vals[2])
    assert np.isclose(vals[1], vals[3])


def test_gaspari_cohn_monotonically_decreasing_on_support():
    z = np.linspace(0, 2, 50)
    vals = gaspari_cohn(z)
    assert np.all(np.diff(vals) <= 1e-8)


def test_cyclic_distance_wraps_around():
    K = 10
    assert cyclic_distance(np.array([0]), np.array([9]), K)[0] == 1
    assert cyclic_distance(np.array([0]), np.array([5]), K)[0] == 5


def test_localization_matrix_shapes_and_self_correlation():
    K = 40
    obs_indices = np.arange(0, K, 2)
    loc_xy, loc_yy = localization_matrix(K, obs_indices, radius=4.0)
    assert loc_xy.shape == (K, len(obs_indices))
    assert loc_yy.shape == (len(obs_indices), len(obs_indices))
    assert np.allclose(np.diag(loc_yy), 1.0)  # an observation is fully correlated with itself


def test_localization_weight_decreases_with_distance():
    K = 40
    obs_indices = np.array([0])
    loc_xy, _ = localization_matrix(K, obs_indices, radius=4.0)
    # weight at state index 0 (distance 0) should exceed weight at index 5 (distance 5)
    assert loc_xy[0, 0] > loc_xy[5, 0]
    assert loc_xy[20, 0] == 0.0  # far side of the ring, beyond 2*radius=8
