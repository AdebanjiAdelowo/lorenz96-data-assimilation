"""Quantitative diagnostics of chaotic behaviour: perturbation growth and the largest Lyapunov
exponent, via the classical Benettin et al. (1980) renormalization algorithm ("Lyapunov Characteristic
Exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of
them. Part 1: Theory", Meccanica 15).

## Why this matters for data assimilation

If nearby trajectories of the "truth" system separate exponentially fast (a positive largest Lyapunov
exponent), a small initial-condition error grows without bound under free (unassimilated) forecasting
-- this is the precise mathematical reason sequential data assimilation is needed at all: periodic
re-anchoring of the forecast to new observations is what prevents this exponential error growth from
making long-range forecasts useless. Section "Free-forecast baseline" in the README demonstrates this
growth directly for the L96 system; this module quantifies its rate.

## Benettin algorithm (largest Lyapunov exponent)

Integrate a reference trajectory x(t) and a perturbed trajectory x(t) + delta(t), delta(0) = delta_0
(a small, fixed-norm perturbation). Over each renormalization interval of length tau:

    delta(t+tau) = phi_tau(x(t) + delta(t)) - phi_tau(x(t))     (phi_tau = the flow map)

then RESCALE delta back to norm |delta_0| before continuing (preventing the perturbed trajectory from
diverging arbitrarily far and leaving the locally-linear regime the Lyapunov exponent characterises),
accumulating

    lambda_1 ~= (1 / (N*tau)) * sum_{n=1}^{N} log( |delta_n_before_rescale| / |delta_0| )

which converges to the largest Lyapunov exponent as N -> infinity (Oseledets' theorem;
Benettin et al. 1980). Implemented in `lyapunov_exponent_benettin`, with the running estimate's
convergence checked directly (`scripts/estimate_lyapunov.py` plots it, and
`tests/test_chaos.py::test_lyapunov_estimate_is_positive_and_roughly_stable` checks it stabilises to a
consistent order of magnitude across two independent renormalization intervals) rather than trusting
a single point estimate.
"""
from __future__ import annotations

import numpy as np

from .l96 import L96Params, integrate


def perturbation_growth(x0: np.ndarray, params: L96Params, delta0: np.ndarray, t_end: float,
                         save_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the reference trajectory from x0 and a perturbed trajectory from x0+delta0, over
    t_end (no renormalization -- this is the raw, unbounded-growth diagnostic showing exponential
    divergence until nonlinear saturation, distinct from the Lyapunov-exponent estimate below).
    Returns (t, ||x_pert(t) - x_true(t)||)."""
    n_steps = int(round(t_end / params.dt))
    t, x_ref = integrate(x0, params, n_steps, save_every)
    _, x_pert = integrate(x0 + delta0, params, n_steps, save_every)
    sep = np.linalg.norm(x_pert - x_ref, axis=1)
    return t, sep


def lyapunov_exponent_benettin(x0: np.ndarray, params: L96Params, t_total: float, tau: float,
                                delta0_norm: float = 1e-8,
                                seed: int = 0) -> tuple[float, np.ndarray, np.ndarray]:
    """Benettin et al. (1980) largest-Lyapunov-exponent estimate. Returns
    (lambda_1_estimate, running_estimate_over_renormalization_steps, times_at_each_renormalization).
    """
    rng = np.random.default_rng(seed)
    K = params.K
    n_steps_per_tau = int(round(tau / params.dt))
    n_renorm = int(round(t_total / tau))

    direction = rng.standard_normal(K)
    direction /= np.linalg.norm(direction)
    delta = delta0_norm * direction

    x = x0.copy()
    log_growth_sum = 0.0
    running = []
    times = []

    for n in range(n_renorm):
        _, x_next = integrate(x, params, n_steps_per_tau, save_every=n_steps_per_tau)
        _, x_pert_next = integrate(x + delta, params, n_steps_per_tau, save_every=n_steps_per_tau)
        x_new = x_next[-1]
        delta_grown = x_pert_next[-1] - x_new

        growth = np.linalg.norm(delta_grown) / delta0_norm
        log_growth_sum += np.log(growth)

        x = x_new
        delta = delta0_norm * delta_grown / np.linalg.norm(delta_grown)

        elapsed = (n + 1) * tau
        running.append(log_growth_sum / elapsed)
        times.append(elapsed)

    lambda_1 = log_growth_sum / t_total
    return lambda_1, np.array(running), np.array(times)
