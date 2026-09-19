"""Shared forecast-analysis cycling experiment runner, reused by every study script (ensemble-size,
observation-density/frequency/noise, localization, inflation, repeated-seed, failure-regime) so the
cycling logic itself is written and verified exactly once.

One "cycle" = integrate every ensemble member forward by `obs_interval_steps` L96 RK4 steps (the
forecast), record forecast diagnostics, draw a noisy observation of the truth at that time, perform
one EnKF analysis update, record analysis diagnostics. A single "truth" trajectory (post-spin-up, so
it starts already on the chaotic attractor -- see src/l96.py:spin_up) is integrated once, continuously,
for the whole experiment; observations are drawn from it at each cycle boundary. A separate, single
"free forecast" trajectory (no assimilation, ever) is integrated from the SAME initial ensemble mean
for the whole experiment, as the baseline free-forecast-error-growth comparison (brief section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .l96 import L96Params, integrate, integrate_ensemble, spin_up
from .enkf import enkf_analysis, ensemble_mean_anomalies, ensemble_covariance
from .observations import ObservationOperator, generate_observation


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ensemble_spread(X: np.ndarray) -> float:
    """sqrt(trace(P)/K) -- the standard scalar ensemble-spread diagnostic (brief section 13)."""
    K = X.shape[0]
    P = ensemble_covariance(X)
    return float(np.sqrt(np.trace(P) / K))


@dataclass
class CycleResult:
    t: np.ndarray  # (n_cycles+1,)
    truth: np.ndarray  # (n_cycles+1, K)
    forecast_mean: np.ndarray  # (n_cycles+1, K)
    analysis_mean: np.ndarray  # (n_cycles+1, K)
    forecast_rmse: np.ndarray  # (n_cycles+1,)
    analysis_rmse: np.ndarray  # (n_cycles+1,)
    forecast_spread: np.ndarray  # (n_cycles+1,)
    analysis_spread: np.ndarray  # (n_cycles+1,)
    free_forecast: np.ndarray  # (n_cycles+1, K)
    free_forecast_rmse: np.ndarray  # (n_cycles+1,)
    obs: list = field(default_factory=list)  # list of (m,) observation vectors, one per cycle
    innovations: list = field(default_factory=list)  # list of (m,) y - H@forecast_mean, per cycle
    diverged: bool = False  # True if analysis RMSE ever exceeds a large threshold
    forecast_ensembles: list = field(default_factory=list)  # list of (K, Ne), only if requested


DIVERGENCE_RMSE_THRESHOLD = 50.0  # well above the L96 attractor's O(1-10) scale


def run_assimilation_experiment(
    params: L96Params, obs_op: ObservationOperator, Ne: int, obs_interval_steps: int, n_cycles: int,
    ens_init_std: float, seed: int, inflation: float = 1.0,
    loc_xy: np.ndarray | None = None, loc_yy: np.ndarray | None = None,
    t_spinup: float = 20.0, save_ensembles: bool = False,
) -> CycleResult:
    rng = np.random.default_rng(seed)
    K = params.K

    x0_generic = np.full(K, params.F)
    x0_generic[0] += 0.01
    truth0 = spin_up(x0_generic, params, t_spinup)

    ens0 = truth0[:, None] + ens_init_std * rng.standard_normal((K, Ne))
    free0 = truth0 + ens_init_std * rng.standard_normal(K)

    t_list = [0.0]
    truth_list = [truth0.copy()]
    fmean_list = [ens0.mean(axis=1)]
    amean_list = [ens0.mean(axis=1)]
    frmse_list = [rmse(fmean_list[0], truth0)]
    armse_list = [rmse(amean_list[0], truth0)]
    fspread_list = [ensemble_spread(ens0)]
    aspread_list = [ensemble_spread(ens0)]
    free_list = [free0.copy()]
    free_rmse_list = [rmse(free0, truth0)]

    X = ens0
    truth = truth0
    free = free0
    t = 0.0
    diverged = False

    obs_list, innov_list, forecast_ens_list = [], [], []

    for cycle in range(n_cycles):
        _, truth_traj = integrate(truth, params, obs_interval_steps, save_every=obs_interval_steps)
        truth = truth_traj[-1]

        X = integrate_ensemble(X, params, obs_interval_steps)
        if save_ensembles:
            forecast_ens_list.append(X.copy())

        _, free_traj = integrate(free, params, obs_interval_steps, save_every=obs_interval_steps)
        free = free_traj[-1]

        t += obs_interval_steps * params.dt

        f_mean = X.mean(axis=1)
        f_rmse = rmse(f_mean, truth)
        f_spread = ensemble_spread(X)

        y = generate_observation(truth, obs_op, rng)
        innovation = y - obs_op.H @ f_mean

        res = enkf_analysis(X, y, obs_op, rng, inflation=inflation, loc_xy=loc_xy, loc_yy=loc_yy)
        X = res.X_a
        a_mean = X.mean(axis=1)
        a_rmse = rmse(a_mean, truth)
        a_spread = ensemble_spread(X)

        t_list.append(t)
        truth_list.append(truth.copy())
        fmean_list.append(f_mean)
        amean_list.append(a_mean)
        frmse_list.append(f_rmse)
        armse_list.append(a_rmse)
        fspread_list.append(f_spread)
        aspread_list.append(a_spread)
        free_list.append(free.copy())
        free_rmse_list.append(rmse(free, truth))
        obs_list.append(y)
        innov_list.append(innovation)

        if a_rmse > DIVERGENCE_RMSE_THRESHOLD or not np.all(np.isfinite(a_mean)):
            diverged = True
            break

    return CycleResult(
        t=np.array(t_list), truth=np.array(truth_list),
        forecast_mean=np.array(fmean_list), analysis_mean=np.array(amean_list),
        forecast_rmse=np.array(frmse_list), analysis_rmse=np.array(armse_list),
        forecast_spread=np.array(fspread_list), analysis_spread=np.array(aspread_list),
        free_forecast=np.array(free_list), free_forecast_rmse=np.array(free_rmse_list),
        obs=obs_list, innovations=innov_list, diverged=diverged,
        forecast_ensembles=forecast_ens_list,
    )


def rank_histogram(result: CycleResult, burn_in_frac: float = 0.2) -> np.ndarray:
    """Talagrand (rank) histogram (brief section 18): for each post-transient cycle and each state
    component, the rank of the truth value among its (sorted) FORECAST ensemble members -- a flat
    histogram is CONSISTENT WITH calibration, a U-shape indicates under-dispersion (truth too often
    falls outside the ensemble's range), a hump indicates over-dispersion. This is a diagnostic, not
    proof of calibration (a flat histogram is necessary, not sufficient, for a correctly specified
    filter -- e.g. it cannot detect a systematic bias shared by truth and ensemble). Requires
    `run_assimilation_experiment(..., save_ensembles=True)`."""
    if not result.forecast_ensembles:
        raise ValueError("rank_histogram requires forecast_ensembles (run with save_ensembles=True)")
    n = len(result.forecast_ensembles)
    start = int(round(burn_in_frac * n))
    Ne = result.forecast_ensembles[0].shape[1]
    ranks = []
    for i in range(start, n):
        X = result.forecast_ensembles[i]  # (K, Ne)
        truth_i = result.truth[i + 1]  # truth AFTER this cycle's forecast step
        for k in range(X.shape[0]):
            ranks.append(int(np.searchsorted(np.sort(X[k, :]), truth_i[k])))
    return np.array(ranks)


def time_averaged_rmse(result: CycleResult, burn_in_frac: float = 0.2) -> tuple[float, float]:
    """Post-spin-up time-averaged (analysis, forecast) RMSE, discarding the first `burn_in_frac`
    fraction of cycles as assimilation transient (brief section 8: report a time-averaged number,
    not one visually attractive interval)."""
    n = len(result.analysis_rmse)
    start = int(round(burn_in_frac * n))
    return float(np.mean(result.analysis_rmse[start:])), float(np.mean(result.forecast_rmse[start:]))


@dataclass
class SeedStats:
    mean: float
    std: float
    values: np.ndarray


def repeated_experiment(seeds: list[int], burn_in_frac: float = 0.2, **kwargs) -> dict:
    """Run `run_assimilation_experiment(**kwargs, seed=s)` for every s in `seeds` (brief section 17:
    key comparisons must not rest on one seed), and summarise time-averaged analysis/forecast RMSE and
    analysis spread as (mean, std) across seeds. `kwargs` must not include `seed`. Returns a dict with
    SeedStats for 'analysis_rmse', 'forecast_rmse', 'analysis_spread', plus 'diverged_fraction' and
    the raw per-seed CycleResults for further inspection (e.g. rank histograms)."""
    a_rmses, f_rmses, spreads, results = [], [], [], []
    for s in seeds:
        res = run_assimilation_experiment(seed=s, **kwargs)
        a_rmse, f_rmse = time_averaged_rmse(res, burn_in_frac)
        n = len(res.analysis_spread)
        start = int(round(burn_in_frac * n))
        a_rmses.append(a_rmse)
        f_rmses.append(f_rmse)
        spreads.append(float(np.mean(res.analysis_spread[start:])))
        results.append(res)

    def stats(vals):
        arr = np.array(vals)
        return SeedStats(mean=float(arr.mean()), std=float(arr.std()), values=arr)

    return {
        "analysis_rmse": stats(a_rmses),
        "forecast_rmse": stats(f_rmses),
        "analysis_spread": stats(spreads),
        "diverged_fraction": float(np.mean([r.diverged for r in results])),
        "results": results,
    }
