"""Observation model: y_k = H x_k + eps_k, eps_k ~ N(0, R).

H selects (and, in principle, could linearly combine) a subset of the K state components; only
DIRECT, possibly-sparse observation of individual state components is used in this project (H rows
are standard basis vectors), matching the classical L96 EnKF literature (Evensen 2009; Kalnay 2003).
R is diagonal (independent observation errors), the standard assumption absent evidence of correlated
instrument error. The filter NEVER receives x_true directly -- only y (and H, R) -- checked explicitly
in tests/test_observations.py and structurally guaranteed by the EnKF API in src/enkf.py, which takes
only (ensemble, y, H, R) as arguments, with no path for the true state to enter the analysis step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObservationOperator:
    H: np.ndarray  # (m, K)
    obs_indices: np.ndarray  # (m,) which state components are observed
    R_diag: np.ndarray  # (m,) observation-error variances (R = diag(R_diag))

    @property
    def m(self) -> int:
        return self.H.shape[0]

    @property
    def R(self) -> np.ndarray:
        return np.diag(self.R_diag)


def make_observation_operator(K: int, stride: int = 1, obs_std: float = 1.0) -> ObservationOperator:
    """Observe every `stride`-th state component (stride=1: fully observed system), each with
    independent Gaussian noise of standard deviation obs_std."""
    obs_indices = np.arange(0, K, stride)
    m = len(obs_indices)
    H = np.zeros((m, K))
    H[np.arange(m), obs_indices] = 1.0
    R_diag = np.full(m, obs_std ** 2)
    return ObservationOperator(H=H, obs_indices=obs_indices, R_diag=R_diag)


def generate_observation(x_true: np.ndarray, obs_op: ObservationOperator,
                          rng: np.random.Generator) -> np.ndarray:
    """y = H x_true + eps, eps_i ~ N(0, R_diag[i]) independently."""
    clean = obs_op.H @ x_true
    noise = rng.normal(0.0, np.sqrt(obs_op.R_diag))
    return clean + noise
