"""Exact Kalman filter for a linear-Gaussian state-space model

    x_{k+1} = A x_k + w_k,   w_k ~ N(0, Q)
    y_k     = H x_k + v_k,   v_k ~ N(0, R)

used ONLY to independently validate the EnKF implementation (src/enkf.py) on a problem with a known
exact posterior mean/covariance (the standard textbook Kalman-filter recursion), before the EnKF is
trusted on the nonlinear Lorenz-96 system -- the data-assimilation analogue of the analytical-
posterior sampler check and the Taylor-remainder gradient checks used elsewhere in this portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanState:
    mean: np.ndarray  # (n,)
    cov: np.ndarray  # (n, n)


def kf_predict(state: KalmanState, A: np.ndarray, Q: np.ndarray) -> KalmanState:
    mean_f = A @ state.mean
    cov_f = A @ state.cov @ A.T + Q
    return KalmanState(mean=mean_f, cov=cov_f)


def kf_update(state: KalmanState, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> KalmanState:
    S = H @ state.cov @ H.T + R
    K = state.cov @ H.T @ np.linalg.inv(S)
    mean_a = state.mean + K @ (y - H @ state.mean)
    n = state.cov.shape[0]
    I = np.eye(n)
    cov_a = (I - K @ H) @ state.cov
    return KalmanState(mean=mean_a, cov=cov_a)


def run_kalman_filter(x0: KalmanState, A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray,
                       observations: np.ndarray) -> list[KalmanState]:
    """observations: (n_steps, m). Returns the list of ANALYSIS states, one per observation, each
    obtained by predict-then-update from the previous analysis (x0 is the prior before the first
    predict)."""
    states = []
    state = x0
    for y in observations:
        state = kf_predict(state, A, Q)
        state = kf_update(state, y, H, R)
        states.append(state)
    return states
