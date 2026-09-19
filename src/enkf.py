"""Stochastic Ensemble Kalman Filter (EnKF) with perturbed observations (Evensen, 1994, "Sequential
data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast
error statistics", J. Geophys. Res. 99(C5); Burgers, van Leeuwen & Evensen, 1998, "Analysis scheme in
the ensemble Kalman filter", Mon. Weather Rev. 126, for the perturbed-observations correction that
makes the update statistically consistent).

## Why stochastic EnKF (over a deterministic square-root filter)

Two standard families exist: stochastic EnKF (perturb each ensemble member's observation
independently before the update) and deterministic ensemble square-root filters (e.g. ETKF, EnSRF,
which update the ensemble without any observation perturbation). This project uses the STOCHASTIC
formulation because (a) it is the original, most directly "from the mathematics" formulation
(Evensen 1994) and the natural analogue of the perturbed-observation idea used nowhere else in this
project, giving a genuinely distinct sequential-Bayesian-estimation methodology; and (b) its own
Monte Carlo (observation-perturbation) sampling noise is an explicit, controllable, and honestly
reportable source of variability -- exactly the kind of thing this project's repeated-seed studies
(section 17 of the brief) are designed to characterise. All random draws (initial ensemble, model
truth noise where used, and the per-member perturbed observations) are made through an explicitly
passed `numpy.random.Generator`, seeded and recorded in every experiment script, so Monte Carlo
variability is never silently mixed into "the" result.

## Ensemble representation and the analysis update

Forecast ensemble $X_f \\in \\mathbb{R}^{K\\times N_e}$ ($N_e$ members, each a $K$-vector), mean
$\\bar x_f = \\frac1{N_e}\\sum_i x_f^i$, anomalies $X'_f = X_f - \\bar x_f\\mathbf 1^T$. The ensemble
estimate of the forecast covariance is

$$P_f \\approx \\frac{1}{N_e-1} X'_f {X'_f}^T,$$

never formed explicitly here (an $O(K^2)$ intermediate we avoid): instead we work with the
observation-space anomalies $Y'_f = H X'_f \\in \\mathbb R^{m\\times N_e}$, so that

$$P_f H^T \\approx \\tfrac{1}{N_e-1} X'_f {Y'_f}^T, \\qquad H P_f H^T \\approx \\tfrac{1}{N_e-1} Y'_f {Y'_f}^T,$$

and the ensemble Kalman gain is

$$K = (P_f H^T)\\,\\big[(H P_f H^T) + R\\big]^{-1}.$$

For the STOCHASTIC update, each member's observation is independently perturbed,
$y^i = y + \\varepsilon^i$, $\\varepsilon^i\\sim N(0,R)$ (this randomised-observation step is exactly
what makes the ensemble-updated sample covariance an unbiased estimator of the Kalman-filter
posterior covariance in expectation -- Burgers et al. 1998; omitting it, i.e. updating every member
toward the SAME unperturbed y, systematically underestimates the analysis spread), and

$$x_a^i = x_f^i + K\\,(y^i - H x_f^i).$$

## Covariance localization and inflation

Both are OPTIONAL, Hadamard (element-wise) modifications applied ONLY if the baseline experiments
demonstrate a need (see README "Localization" / "Inflation" -- neither is applied by default):

- **Localization** (Gaspari & Cohn, 1999): replaces $P_fH^T$, $HP_fH^T$ with
  $\\rho_{xy}\\odot(P_fH^T)$, $\\rho_{yy}\\odot(HP_fH^T)$, $\\rho$ a compactly-supported correlation
  function of cyclic-ring distance between state/observation locations (`src/localization.py`),
  damping the spurious long-range sample correlations a small ensemble produces in a
  higher-dimensional state space.
- **Inflation**: forecast anomalies scaled by $\\sqrt{\\lambda}$, $\\lambda\\ge1$, before computing the
  gain, counteracting ensemble-collapse/under-dispersion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observations import ObservationOperator


def ensemble_mean_anomalies(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X: (K, Ne). Returns (mean (K,), anomalies (K, Ne))."""
    mean = X.mean(axis=1)
    anomalies = X - mean[:, None]
    return mean, anomalies


def ensemble_covariance(X: np.ndarray) -> np.ndarray:
    """Explicit (K, K) sample covariance -- only for diagnostics (spread, localization comparison
    plots); never used inside the analysis update itself (see module docstring)."""
    _, A = ensemble_mean_anomalies(X)
    Ne = X.shape[1]
    return (A @ A.T) / (Ne - 1)


@dataclass
class EnKFResult:
    X_a: np.ndarray  # (K, Ne) analysis ensemble
    gain: np.ndarray  # (K, m) Kalman gain used


def enkf_analysis(X_f: np.ndarray, y: np.ndarray, obs_op: ObservationOperator,
                   rng: np.random.Generator, inflation: float = 1.0,
                   loc_xy: np.ndarray | None = None, loc_yy: np.ndarray | None = None) -> EnKFResult:
    """Stochastic EnKF analysis update (see module docstring). X_f: (K, Ne) forecast ensemble.
    inflation: multiplicative variance inflation factor lambda (anomalies scaled by sqrt(lambda));
    1.0 = no inflation. loc_xy (K, m), loc_yy (m, m): optional Gaspari-Cohn localization matrices
    (see src/localization.py); None = no localization."""
    K, Ne = X_f.shape
    H, R = obs_op.H, obs_op.R
    m = obs_op.m

    mean_f, A = ensemble_mean_anomalies(X_f)
    if inflation != 1.0:
        A = np.sqrt(inflation) * A
        X_f = mean_f[:, None] + A

    Y = H @ A  # (m, Ne): observation-space anomalies

    PfHt = (A @ Y.T) / (Ne - 1)  # (K, m)
    HPfHt = (Y @ Y.T) / (Ne - 1)  # (m, m)

    if loc_xy is not None:
        PfHt = loc_xy * PfHt
    if loc_yy is not None:
        HPfHt = loc_yy * HPfHt

    gain = PfHt @ np.linalg.inv(HPfHt + R)  # (K, m)

    X_a = np.empty_like(X_f)
    for i in range(Ne):
        y_pert = y + rng.normal(0.0, np.sqrt(obs_op.R_diag))
        innovation = y_pert - H @ X_f[:, i]
        X_a[:, i] = X_f[:, i] + gain @ innovation

    return EnKFResult(X_a=X_a, gain=gain)
