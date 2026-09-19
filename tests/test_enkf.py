import numpy as np

from src.enkf import ensemble_mean_anomalies, ensemble_covariance, enkf_analysis
from src.observations import ObservationOperator, make_observation_operator


def test_ensemble_mean_and_anomalies():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    mean, A = ensemble_mean_anomalies(X)
    assert np.allclose(mean, [2.0, 6.0])
    assert np.allclose(A, X - mean[:, None])
    assert np.allclose(A.sum(axis=1), 0.0)


def test_ensemble_covariance_matches_numpy_cov():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 200))
    C = ensemble_covariance(X)
    assert np.allclose(C, np.cov(X), atol=1e-10)


def test_enkf_gain_shape_and_finiteness():
    K, Ne = 10, 30
    rng = np.random.default_rng(1)
    X_f = rng.standard_normal((K, Ne)) + 8.0
    obs_op = make_observation_operator(K, stride=2, obs_std=1.0)
    y = obs_op.H @ np.full(K, 8.0)
    res = enkf_analysis(X_f, y, obs_op, rng)
    assert res.gain.shape == (K, obs_op.m)
    assert np.all(np.isfinite(res.gain))
    assert np.all(np.isfinite(res.X_a))


def test_enkf_analysis_moves_ensemble_mean_toward_observation():
    """A basic sanity check: if the forecast ensemble mean disagrees with a (large, unambiguous)
    observed signal, the analysis mean should move toward it at the observed locations."""
    K = 10
    rng = np.random.default_rng(2)
    X_f = 8.0 + 0.1 * rng.standard_normal((K, 50))  # tightly clustered around 8
    obs_op = make_observation_operator(K, stride=1, obs_std=0.01)  # very confident observation
    y = np.full(K, 20.0)  # observation says the truth is near 20, far from the forecast
    res = enkf_analysis(X_f, y, obs_op, rng)
    analysis_mean = res.X_a.mean(axis=1)
    forecast_mean = X_f.mean(axis=1)
    assert np.all(np.abs(analysis_mean - 20.0) < np.abs(forecast_mean - 20.0))


def test_inflation_increases_analysis_spread():
    K, Ne = 8, 30
    rng = np.random.default_rng(3)
    X_f = rng.standard_normal((K, Ne)) + 8.0
    obs_op = make_observation_operator(K, stride=1, obs_std=1.0)
    y = np.full(K, 8.0)
    res_no_infl = enkf_analysis(X_f.copy(), y, obs_op, np.random.default_rng(5), inflation=1.0)
    res_infl = enkf_analysis(X_f.copy(), y, obs_op, np.random.default_rng(5), inflation=2.0)
    spread_no_infl = np.sqrt(np.trace(ensemble_covariance(res_no_infl.X_a)) / K)
    spread_infl = np.sqrt(np.trace(ensemble_covariance(res_infl.X_a)) / K)
    assert spread_infl > spread_no_infl


def test_enkf_reproducible_given_seed():
    K, Ne = 10, 20
    X_f = np.full(K, 8.0)[:, None] + np.random.default_rng(0).standard_normal((K, Ne))
    obs_op = make_observation_operator(K, stride=2, obs_std=1.0)
    y = obs_op.H @ np.full(K, 8.0)
    res1 = enkf_analysis(X_f.copy(), y, obs_op, np.random.default_rng(42))
    res2 = enkf_analysis(X_f.copy(), y, obs_op, np.random.default_rng(42))
    assert np.array_equal(res1.X_a, res2.X_a)


def test_observation_operator_selects_correct_indices():
    obs_op = make_observation_operator(K=10, stride=3, obs_std=0.5)
    assert np.array_equal(obs_op.obs_indices, [0, 3, 6, 9])
    x = np.arange(10, dtype=float)
    assert np.allclose(obs_op.H @ x, [0.0, 3.0, 6.0, 9.0])
    assert np.allclose(obs_op.R_diag, 0.25)


def test_filter_never_receives_true_state_directly():
    """Structural check: enkf_analysis's signature takes no argument through which x_true could
    enter -- only the forecast ensemble, the (already-generated) observation, the observation
    operator, and an RNG."""
    import inspect
    sig = inspect.signature(enkf_analysis)
    params = list(sig.parameters)
    assert params == ["X_f", "y", "obs_op", "rng", "inflation", "loc_xy", "loc_yy"]
