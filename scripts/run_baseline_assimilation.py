"""Baseline EnKF assimilation experiment (brief section 8): partial/full noisy observations,
perturbed initial ensemble, repeated forecast-analysis cycles. Deliberately uses a MODEST ensemble
size (Ne=20 for a K=40 system, no localization/inflation) rather than a pre-tuned "success"
configuration -- this both answers "does assimilation help at all" (yes, relative to the free
forecast) and honestly exposes the finite-ensemble sampling problem that motivates the localization/
inflation studies run later.

Usage: python scripts/run_baseline_assimilation.py [--config smoke|local|full]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.l96 import L96Params
from src.observations import make_observation_operator
from src.experiment import run_assimilation_experiment, time_averaged_rmse

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, b = cfg["base"], cfg["baseline"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=b["stride"], obs_std=b["obs_std"])

    res = run_assimilation_experiment(
        params, obs_op, Ne=b["Ne"], obs_interval_steps=b["obs_interval_steps"],
        n_cycles=b["n_cycles"], ens_init_std=b["ens_init_std"], seed=b["seed"],
        t_spinup=base["t_spinup"],
    )
    a_rmse, f_rmse = time_averaged_rmse(res)

    lines = [
        f"Baseline EnKF assimilation ({args.config} config): K={base['K']}, F={base['F']}, "
        f"Ne={b['Ne']}, obs stride={b['stride']} (m={obs_op.m}/{base['K']} observed), "
        f"obs_std={b['obs_std']}, obs interval={b['obs_interval_steps']*base['dt']:.3f} time units, "
        f"{b['n_cycles']} cycles, seed={b['seed']}",
        f"diverged: {res.diverged}",
        f"time-averaged (post-transient) analysis RMSE: {a_rmse:.4f}",
        f"time-averaged (post-transient) forecast RMSE: {f_rmse:.4f}",
        f"final free-forecast RMSE (no assimilation, same run length): {res.free_forecast_rmse[-1]:.4f}",
        f"time-averaged analysis spread: {np.mean(res.analysis_spread[len(res.analysis_spread)//5:]):.4f}",
        f"analysis RMSE / spread ratio (post-transient): "
        f"{a_rmse / np.mean(res.analysis_spread[len(res.analysis_spread)//5:]):.2f} "
        "(1.0 would indicate a well-calibrated ensemble in this crude sense; much greater than 1 "
        "indicates under-dispersion -- see README 'Ensemble spread')",
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"baseline_assimilation_{args.config}.txt").write_text("\n".join(lines) + "\n")
    np.savez_compressed(RESULTS_DIR / f"baseline_assimilation_{args.config}.npz",
                         t=res.t, truth=res.truth, forecast_mean=res.forecast_mean,
                         analysis_mean=res.analysis_mean, analysis_rmse=res.analysis_rmse,
                         forecast_rmse=res.forecast_rmse, analysis_spread=res.analysis_spread,
                         forecast_spread=res.forecast_spread, free_forecast_rmse=res.free_forecast_rmse)
    for line in lines:
        print(line)

    # --- figures --------------------------------------------------------------------------------
    fig1, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for comp, ax in zip([0, 10, 20], axes):
        ax.plot(res.t, res.truth[:, comp], "k-", label="truth", lw=1.5)
        ax.plot(res.t, res.analysis_mean[:, comp], "C0--", label="EnKF analysis mean", lw=1.0)
        if comp in obs_op.obs_indices:
            j = int(np.where(obs_op.obs_indices == comp)[0][0])
            obs_vals = [o[j] for o in res.obs]
            ax.plot(res.t[1:], obs_vals, "C1.", ms=3, alpha=0.5, label="observations")
        ax.set_ylabel(f"$x_{{{comp}}}$")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("t")
    fig1.suptitle(f"Truth / observations / EnKF analysis ({args.config} config)")
    fig1.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig1.savefig(FIG_DIR / f"baseline_components_{args.config}.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(res.t, res.analysis_rmse, label="EnKF analysis RMSE")
    ax.semilogy(res.t, res.free_forecast_rmse, "--", label="free-forecast RMSE (no assimilation)")
    ax.set_xlabel("t")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE vs. time: EnKF analysis vs. free forecast ({args.config})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / f"baseline_rmse_vs_time_{args.config}.png", dpi=150)

    fig3, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res.t, res.analysis_rmse, label="analysis RMSE")
    ax.plot(res.t, res.analysis_spread, label="analysis spread")
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(f"Spread vs. RMSE ({args.config}, Ne={b['Ne']}, no localization/inflation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / f"baseline_spread_vs_rmse_{args.config}.png", dpi=150)

    print(f"\nWrote figures/baseline_{{components,rmse_vs_time,spread_vs_rmse}}_{args.config}.png "
          f"and results/baseline_assimilation_{args.config}.{{txt,npz}}")


if __name__ == "__main__":
    main()
