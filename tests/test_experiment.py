import numpy as np

from src.l96 import L96Params
from src.observations import make_observation_operator
from src.experiment import (
    run_assimilation_experiment, time_averaged_rmse, repeated_experiment, rmse, ensemble_spread,
    rank_histogram,
)


def _base_kwargs(K=20, Ne=10, n_cycles=15):
    params = L96Params(K=K, F=8.0, dt=0.01)
    obs_op = make_observation_operator(K, stride=1, obs_std=1.0)
    return dict(params=params, obs_op=obs_op, Ne=Ne, obs_interval_steps=10, n_cycles=n_cycles,
                ens_init_std=1.0, t_spinup=5.0)


def test_rmse_zero_for_identical_arrays():
    x = np.array([1.0, 2.0, 3.0])
    assert rmse(x, x) == 0.0


def test_ensemble_spread_matches_trace_formula():
    K = 5
    rng = np.random.default_rng(0)
    X = rng.standard_normal((K, 100))
    from src.enkf import ensemble_covariance
    expected = np.sqrt(np.trace(ensemble_covariance(X)) / K)
    assert np.isclose(ensemble_spread(X), expected)


def test_experiment_output_shapes_consistent():
    res = run_assimilation_experiment(seed=0, **_base_kwargs())
    n = len(res.t)
    assert res.truth.shape == (n, 20)
    assert res.forecast_mean.shape == (n, 20)
    assert res.analysis_mean.shape == (n, 20)
    assert len(res.analysis_rmse) == n
    assert len(res.obs) == n - 1


def test_analysis_rmse_beats_free_forecast_rmse_in_a_working_configuration():
    """A reasonably-sized, fully-observed ensemble should track the truth much better than an
    unassimilated free forecast over many cycles."""
    kwargs = _base_kwargs(K=20, Ne=20, n_cycles=60)
    res = run_assimilation_experiment(seed=1, **kwargs)
    a_rmse, _ = time_averaged_rmse(res)
    assert a_rmse < res.free_forecast_rmse[-1]


def test_repeated_experiment_returns_stats_over_seeds():
    kwargs = _base_kwargs(K=20, Ne=15, n_cycles=20)
    out = repeated_experiment(seeds=[0, 1, 2], **kwargs)
    assert out["analysis_rmse"].values.shape == (3,)
    assert out["analysis_rmse"].mean > 0
    assert out["analysis_rmse"].std >= 0
    assert len(out["results"]) == 3


def test_rank_histogram_raises_without_saved_ensembles():
    res = run_assimilation_experiment(seed=0, **_base_kwargs())
    try:
        rank_histogram(res)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rank_histogram_values_within_valid_range():
    kwargs = _base_kwargs(K=10, Ne=8, n_cycles=15)
    res = run_assimilation_experiment(seed=0, save_ensembles=True, **kwargs)
    ranks = rank_histogram(res, burn_in_frac=0.2)
    assert ranks.min() >= 0
    assert ranks.max() <= 8  # Ne=8 -> ranks in [0, Ne]
    assert len(ranks) > 0


def test_experiment_deterministic_given_seed():
    kwargs = _base_kwargs()
    res1 = run_assimilation_experiment(seed=5, **kwargs)
    res2 = run_assimilation_experiment(seed=5, **kwargs)
    assert np.array_equal(res1.analysis_mean, res2.analysis_mean)
    assert np.array_equal(res1.truth, res2.truth)
