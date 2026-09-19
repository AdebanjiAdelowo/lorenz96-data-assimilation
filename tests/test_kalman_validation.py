"""Independent EnKF validation against the exact Kalman filter on a small linear-Gaussian system
(project brief section 7) -- the data-assimilation analogue of the analytical-posterior sampler check
used for the pCN sampler in the companion `darcy-inverse-problem` project."""
import numpy as np

from src.kalman_reference import KalmanState, run_kalman_filter
from src.enkf import enkf_analysis
from src.observations import ObservationOperator


def _linear_system(n=4, seed=0):
    theta = 0.3
    A = 0.95 * np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0.9, 0.05],
        [0, 0, -0.05, 0.9],
    ])
    Q = 0.01 * np.eye(n)
    obs_indices = np.array([0, 2])
    m = len(obs_indices)
    H = np.zeros((m, n))
    H[np.arange(m), obs_indices] = 1.0
    R_diag = np.full(m, 0.1)
    obs_op = ObservationOperator(H=H, obs_indices=obs_indices, R_diag=R_diag)

    rng = np.random.default_rng(seed)
    n_steps = 40
    x_true = rng.standard_normal(n)
    obs_list = []
    for _ in range(n_steps):
        x_true = A @ x_true + rng.multivariate_normal(np.zeros(n), Q)
        y = H @ x_true + rng.normal(0.0, np.sqrt(R_diag))
        obs_list.append(y)
    return A, Q, H, obs_op, np.array(obs_list), n


def _run_enkf(A, Q, obs_op, observations, Ne, n, seed):
    rng = np.random.default_rng(seed)
    x0_mean = np.zeros(n)
    x0_cov = 2.0 * np.eye(n)
    X = x0_mean[:, None] + rng.multivariate_normal(np.zeros(n), x0_cov, size=Ne).T
    ens_means, ens_covs = [], []
    for y in observations:
        for i in range(Ne):
            X[:, i] = A @ X[:, i] + rng.multivariate_normal(np.zeros(n), Q)
        res = enkf_analysis(X, y, obs_op, rng)
        X = res.X_a
        ens_means.append(X.mean(axis=1))
        ens_covs.append(np.cov(X))
    return ens_means, ens_covs


def test_enkf_mean_converges_to_kalman_filter_as_ensemble_grows():
    A, Q, H, obs_op, observations, n = _linear_system()
    kf_states = run_kalman_filter(KalmanState(np.zeros(n), 2.0 * np.eye(n)), A, Q, H, obs_op.R,
                                   observations)

    errors_by_Ne = {}
    for Ne in [10, 100, 2000]:
        ens_means, _ = _run_enkf(A, Q, obs_op, observations, Ne, n, seed=1)
        errs = [np.linalg.norm(ens_means[k] - kf_states[k].mean) for k in range(len(observations))]
        errors_by_Ne[Ne] = np.mean(errs[-10:])  # post-transient average

    assert errors_by_Ne[2000] < errors_by_Ne[100] < errors_by_Ne[10]
    assert errors_by_Ne[2000] < 0.1


def test_enkf_covariance_converges_to_kalman_filter_as_ensemble_grows():
    A, Q, H, obs_op, observations, n = _linear_system()
    kf_states = run_kalman_filter(KalmanState(np.zeros(n), 2.0 * np.eye(n)), A, Q, H, obs_op.R,
                                   observations)

    errors_by_Ne = {}
    for Ne in [10, 100, 2000]:
        _, ens_covs = _run_enkf(A, Q, obs_op, observations, Ne, n, seed=2)
        errs = [np.linalg.norm(ens_covs[k] - kf_states[k].cov) for k in range(len(observations))]
        errors_by_Ne[Ne] = np.mean(errs[-10:])

    assert errors_by_Ne[2000] < errors_by_Ne[10]
    assert errors_by_Ne[2000] < 0.05
